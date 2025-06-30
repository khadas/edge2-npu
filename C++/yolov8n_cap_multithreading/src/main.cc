// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <queue>
#include <thread>
#include <mutex>

#define _BASETSD_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>

#include "RgaUtils.h"
#include "im2d.h"
#include "postprocess.h"
#include "camera_util.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
#define WIDTH  1920
#define HEIGHT 1080
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  	unsigned char* data;
  	int            ret;

  	data = NULL;

  	if (NULL == fp) {
		return NULL;
  	}

  	ret = fseek(fp, ofst, SEEK_SET);
  	if (ret != 0) {
		printf("blob seek failure.\n");
		return NULL;
  	}

  	data = (unsigned char*)malloc(sz);
  	if (data == NULL) {
		printf("buffer malloc failure.\n");
		return NULL;
  	}
  	ret = fread(data, 1, sz, fp);
  	return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  	FILE*          fp;
  	unsigned char* data;

  	fp = fopen(filename, "rb");
  	if (NULL == fp) {
		printf("Open file %s failed.\n", filename);
		return NULL;
  	}

  	fseek(fp, 0, SEEK_END);
  	int size = ftell(fp);

  	data = load_data(fp, 0, size);

  	fclose(fp);

  	*model_size = size;
  	return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  	FILE* fp;
  	fp = fopen(file_name, "w");
  	for (int i = 0; i < element_size; i++) {
		fprintf(fp, "%.6f\n", output[i]);
  	}
  	fclose(fp);
  	return 0;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/

std::mutex input_mtx, output_mtx;
std::queue<cv::Mat> input_queue;
std::queue<cv::Mat> output_queue;
int max_input_queue_size = 5;
int input_queue_num = 0;
int output_queue_num = 0;

void read_image(std::string camera_type, std::string device_number)
{
	int ret;
	
	if (camera_type == "usb") {
		ret = load_usb_camera(device_number, WIDTH, HEIGHT);
	}
	else if (camera_type == "mipi") {
		ret = load_mipi_camera(device_number, WIDTH, HEIGHT);
	}
	
	while (true) {
		cv::Mat image;
		read_usb_frame(&image);
		
		input_mtx.lock();
		if (input_queue.size() < max_input_queue_size) {
			input_queue.push(image.clone());
		}
		else {
			input_queue.pop();
			input_queue.push(image.clone());
		}
		input_mtx.unlock();
	}
}

void infer_model(char* model_name, int thread_id)
{
	int            status     = 0;
  	rknn_context   ctx;
  	size_t         actual_size        = 0;
  	int            img_width          = 0;
  	int            img_height         = 0;
  	int            img_channel        = 0;
  	const float    nms_threshold      = NMS_THRESH;
  	const float    box_conf_threshold = BOX_THRESH;
  	struct timeval start_time, stop_time;
  	int            ret;

  	// init rga context
  	rga_buffer_t src;
  	rga_buffer_t dst;
  	im_rect      src_rect;
  	im_rect      dst_rect;
  	memset(&src_rect, 0, sizeof(src_rect));
  	memset(&dst_rect, 0, sizeof(dst_rect));
  	memset(&src, 0, sizeof(src));
  	memset(&dst, 0, sizeof(dst));
	
	int            model_data_size = 0;
  	unsigned char* model_data      = load_model(model_name, &model_data_size);
  	ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);

  	rknn_sdk_version version;
  	ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));

  	rknn_input_output_num io_num;
  	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

  	rknn_tensor_attr input_attrs[io_num.n_input];
  	memset(input_attrs, 0, sizeof(input_attrs));
  	for (int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
		dump_tensor_attr(&(input_attrs[i]));
  	}

  	rknn_tensor_attr output_attrs[io_num.n_output];
  	memset(output_attrs, 0, sizeof(output_attrs));
  	for (int i = 0; i < io_num.n_output; i++) {
		output_attrs[i].index = i;
		ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
		dump_tensor_attr(&(output_attrs[i]));
  	}

  	int channel = 3;
  	int width   = 0;
  	int height  = 0;
  	if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
		channel = input_attrs[0].dims[1];
		width   = input_attrs[0].dims[2];
		height  = input_attrs[0].dims[3];
  	} else {
		width   = input_attrs[0].dims[1];
		height  = input_attrs[0].dims[2];
		channel = input_attrs[0].dims[3];
  	}

  	//printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  	rknn_input inputs[1];
  	memset(inputs, 0, sizeof(inputs));
  	inputs[0].index        = 0;
  	inputs[0].type         = RKNN_TENSOR_UINT8;
  	inputs[0].size         = width * height * channel;
  	inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	inputs[0].pass_through = 0;

	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 0;
	}
	
	float scale_w, scale_h;
	int resize_w, resize_h, padding;
	if (WIDTH > HEIGHT) {
		scale_w = (float)width / WIDTH;
		scale_h = scale_w;
		resize_w = width;
		resize_h = (int)(resize_w * HEIGHT / WIDTH);
		padding = resize_w - resize_h;
	}
	else {
		scale_h = (float)height / HEIGHT;
		scale_w = scale_h;
		resize_h = height;
		resize_w = (int)(resize_h * WIDTH / HEIGHT);
		padding = resize_h - resize_w;
	}

  	detect_result_group_t detect_result_group;
  	std::vector<float>    out_scales;
  	std::vector<int32_t>  out_zps;
  	char text[256];
  	
  	int tmp, x1, y1, x2, y2, i;
  	cv::Mat img;
  	
  	while (true) {
		input_mtx.lock();
		if (input_queue.empty()) {
			input_mtx.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		cv::Mat orig_img = input_queue.front();
		input_queue.pop();
		tmp = input_queue_num;
		input_queue_num++;
		input_mtx.unlock();
		
		cv::resize(orig_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
		if (WIDTH > HEIGHT) {
			cv::copyMakeBorder(img, img, 0, padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		}
		else{
			cv::copyMakeBorder(img, img, 0, 0, 0, padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		}

		inputs[0].buf = (void*)img.data;
		
		rknn_inputs_set(ctx, io_num.n_input, inputs);

		for (i = 0; i < io_num.n_output; i++) {
			outputs[i].want_float = 0;
		}

		ret = rknn_run(ctx, NULL);
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

		for (i = 0; i < io_num.n_output; ++i) {
			out_scales.push_back(output_attrs[i].scale);
			out_zps.push_back(output_attrs[i].zp);
		}
		post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
				box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

		for (i = 0; i < detect_result_group.count; i++) {
			detect_result_t* det_result = &(detect_result_group.results[i]);
			sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
			//printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
			//		det_result->box.right, det_result->box.bottom, det_result->prop);
			x1 = det_result->box.left;
			y1 = det_result->box.top;
			x2 = det_result->box.right;
			y2 = det_result->box.bottom;
			rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 3);
			putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}
		ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
		
		while (true) {
			output_mtx.lock();
			if (tmp == output_queue_num) {
				output_queue.push(orig_img.clone());
				output_queue_num++;
				output_mtx.unlock();
				break;
			}
			output_mtx.unlock();
		}
	}

	deinitPostProcess();
  	// release
  	ret = rknn_destroy(ctx);

  	if (model_data) {
		free(model_data);
  	}
}

void show_image()
{
	struct timeval start_time, stop_time;
	gettimeofday(&start_time, NULL);
	int n = 0;
	float total_time = 0;
	while (true) {
		output_mtx.lock();
		if (output_queue.empty()) {
			output_mtx.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		cv::Mat image = output_queue.front();
		output_queue.pop();
		cv::imshow("Image Window", image);
		cv::waitKey(1);
		gettimeofday(&stop_time, NULL);
		total_time += (__get_us(stop_time) - __get_us(start_time)) / 1000;
		n++;
		if (n == 10)
		{
			printf("average time : %f ms\n", (total_time / 10));
			total_time = 0;
			n = 0;
		}
		gettimeofday(&start_time, NULL);
		output_mtx.unlock();
	}
}

int main(int argc, char** argv)
{
  	int            status     = 0;
  	char*          model_name = NULL;
  	rknn_context   ctx;
  	size_t         actual_size        = 0;
  	int            img_width          = 0;
  	int            img_height         = 0;
  	int            img_channel        = 0;
  	const float    nms_threshold      = NMS_THRESH;
  	const float    box_conf_threshold = BOX_THRESH;
  	struct timeval start_time, stop_time;
  	int            ret;

  	// init rga context
  	rga_buffer_t src;
  	rga_buffer_t dst;
  	im_rect      src_rect;
  	im_rect      dst_rect;
  	memset(&src_rect, 0, sizeof(src_rect));
  	memset(&dst_rect, 0, sizeof(dst_rect));
  	memset(&src, 0, sizeof(src));
  	memset(&dst, 0, sizeof(dst));

  	if (argc != 5) {
		printf("Usage: %s <rknn model> <usb or mipi> <device number> <threads num> \n", argv[0]);
		return -1;
  	}

  	printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  	model_name       = (char*)argv[1];
	std::string camera_type = argv[2];
	std::string device_number = argv[3];
	int thread_num = std::atoi(argv[4]);

	if (camera_type != "usb" && camera_type != "mipi") {
		std::cout << "Unsupport camera type : " << camera_type << " !!!" << std::endl;
	}
	
	std::thread read_image_thread(read_image, camera_type, device_number);
	std::thread show_image_thread(show_image);
	
	std::vector<std::thread> threads;
	for (int i = 0; i < thread_num; i++)
	{
		threads.emplace_back(infer_model, model_name, i);
	}
	
	for (auto& t : threads)
	{
		t.join();
	}
	
	read_image_thread.join();
	show_image_thread.join();

  	return 0;
}
