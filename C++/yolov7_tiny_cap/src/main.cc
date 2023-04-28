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
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
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

  	if (argc != 3) {
		printf("Usage: %s <rknn model> <device number> \n", argv[0]);
		return -1;
  	}

  	printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  	model_name       = (char*)argv[1];

	std::string device_number = argv[2];

	cv::namedWindow("Image Window");

  	/* Create the neural network */
  	printf("Loading mode...\n");
  	int            model_data_size = 0;
  	unsigned char* model_data      = load_model(model_name, &model_data_size);
  	ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}

  	rknn_sdk_version version;
  	ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  	rknn_input_output_num io_num;
  	ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  	rknn_tensor_attr input_attrs[io_num.n_input];
  	memset(input_attrs, 0, sizeof(input_attrs));
  	for (int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret < 0) {
	  		printf("rknn_init error ret=%d\n", ret);
	  		return -1;
		}
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
		printf("model is NCHW input fmt\n");
		channel = input_attrs[0].dims[1];
		width   = input_attrs[0].dims[2];
		height  = input_attrs[0].dims[3];
  	} else {
		printf("model is NHWC input fmt\n");
		width   = input_attrs[0].dims[1];
		height  = input_attrs[0].dims[2];
		channel = input_attrs[0].dims[3];
  	}

  	printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  	rknn_input inputs[1];
  	memset(inputs, 0, sizeof(inputs));
  	inputs[0].index        = 0;
  	inputs[0].type         = RKNN_TENSOR_UINT8;
  	inputs[0].size         = width * height * channel;
  	inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	inputs[0].pass_through = 0;

  	// You may not need resize when src resulotion equals to dst resulotion
  	void* resize_buf = nullptr;
	resize_buf = malloc(height * width * channel);

	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 0;
	}

	cv::Mat orig_img;
	cv::Mat img;
	cv::VideoCapture cap(std::stoi(device_number));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

	if (!cap.isOpened()) {
		printf("capture device failed to open!");
		cap.release();
		exit(-1);
	}

	if (!cap.read(orig_img)) {
		printf("Capture read error");
	}
  	cv::copyMakeBorder(orig_img, img, 0, 840, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  	img_width  = orig_img.cols;
  	img_height = orig_img.rows;
  	printf("img width = %d, img height = %d\n", img_width, img_height);

	// post process
  	float scale_w = (float)width / img_width;
  	float scale_h = scale_w;

  	detect_result_group_t detect_result_group;
  	std::vector<float>    out_scales;
  	std::vector<int32_t>  out_zps;
  	char text[256];

	int x1,y1,x2,y2,i;

	src = wrapbuffer_virtualaddr((void*)img.data, img_width, img_width, RK_FORMAT_RGB_888);
	dst = wrapbuffer_virtualaddr((void*)resize_buf, width, height, RK_FORMAT_RGB_888);

	while(1){

		if (!cap.read(orig_img)) {
			printf("Capture read error");
			break;
		}
		// cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
		cv::copyMakeBorder(orig_img, img, 0, 840, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

		memset(resize_buf, 0x00, height * width * channel);

		imresize(src, dst);
		inputs[0].buf = resize_buf;

		gettimeofday(&start_time, NULL);
		rknn_inputs_set(ctx, io_num.n_input, inputs);

		for (i = 0; i < io_num.n_output; i++) {
			outputs[i].want_float = 0;
		}

		ret = rknn_run(ctx, NULL);
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
		gettimeofday(&stop_time, NULL);
		printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

		for (i = 0; i < io_num.n_output; ++i) {
			out_scales.push_back(output_attrs[i].scale);
			out_zps.push_back(output_attrs[i].zp);
		}
		post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
				box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

		for (i = 0; i < detect_result_group.count; i++) {
			detect_result_t* det_result = &(detect_result_group.results[i]);
			sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
			printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
					det_result->box.right, det_result->box.bottom, det_result->prop);
			x1 = det_result->box.left;
			y1 = det_result->box.top;
			x2 = det_result->box.right;
			y2 = det_result->box.bottom;
			rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
			putText(orig_img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}

		cv::imshow("Image Window",orig_img);
		cv::waitKey(1);

		ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
	}

	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

	deinitPostProcess();
  	// release
  	ret = rknn_destroy(ctx);

  	if (model_data) {
		free(model_data);
  	}

  	if (resize_buf) {
		free(resize_buf);
  	}

  	return 0;
}
