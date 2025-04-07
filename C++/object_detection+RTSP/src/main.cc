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
	const float    nms_threshold      = NMS_THRESH;
	const float    box_conf_threshold = BOX_THRESH;
	if (argc != 4) {
		printf("Usage: %s <rknn model> <device number> <RTSP URL> \n", argv[0]);
		return -1;
	}

	char* model_name = argv[1];
	std::string device_number = argv[2];
	std::string rtsp_url = argv[3];

	// Initialize camera
	cv::VideoCapture cap(std::stoi(device_number));
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

	if (!cap.isOpened()) {
		printf("Failed to open camera!\n");
		return -1;
	}

	// Initialize model
	rknn_context ctx;
	int model_data_size = 0;
	unsigned char* model_data = load_model(model_name, &model_data_size);

	int ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
	if (ret < 0) {
		printf("rknn_init failed! ret=%d\n", ret);
		return -1;
	}

	// FFmpeg command for RTSP streaming
	std::string command = "ffmpeg -y -f rawvideo -pixel_format bgr24 "
							"-video_size 1920x1080 -framerate 30 "
							"-i - -c:v h264_rkmpp -rc_mode AVBR -b:v 20M "
							"-minrate 10M -maxrate 40M -profile:v main -level 5.1 "
							"-f rtsp " + rtsp_url;

	FILE* ffmpeg = popen(command.c_str(), "w");
	if (!ffmpeg) {
		printf("Failed to start FFmpeg!\n");
		return -1;
	}

	// Query model information
	rknn_input_output_num io_num;
	rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));

	rknn_tensor_attr input_attrs[io_num.n_input];
	memset(input_attrs, 0, sizeof(input_attrs));
	for (int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
	}

	rknn_tensor_attr output_attrs[io_num.n_output];
	memset(output_attrs, 0, sizeof(output_attrs));
	for (int i = 0; i < io_num.n_output; i++) {
		output_attrs[i].index = i;
		rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
	}

	int width = input_attrs[0].dims[1];
	int height = input_attrs[0].dims[2];
	int channel = input_attrs[0].dims[3];

	// Initialize input and output buffers
	rknn_input inputs[1];
	memset(inputs, 0, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = width * height * channel;
	inputs[0].fmt = RKNN_TENSOR_NHWC;

	float actual_width = 1920;
	float actual_height = 1080;

	float scale_w = (float)width / actual_width;
	float scale_h = (float)height / actual_height;
	printf("scale_w = %f, scale_h = %f\n", scale_w, scale_h);

	void* resize_buf = malloc(height * width * channel);

	rknn_output outputs[io_num.n_output];
	memset(outputs, 0, sizeof(outputs));
	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 0;
	}

	cv::Mat orig_img, img, resized_img;

	while (cap.read(orig_img)) {
		// Preprocess image
		cv::resize(orig_img, resized_img, cv::Size(width, height));
		memcpy(resize_buf, resized_img.data, width * height * channel);

		// Set input
		inputs[0].buf = resize_buf;
		rknn_inputs_set(ctx, io_num.n_input, inputs);

		// Inference
		struct timeval start_time, stop_time;
		gettimeofday(&start_time, NULL);
		ret = rknn_run(ctx, NULL);
		ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
		gettimeofday(&stop_time, NULL);

		// Detect results
		detect_result_group_t detect_result_group;
		std::vector<float> out_scales;
		std::vector<int32_t> out_zps;

		for (int i = 0; i < io_num.n_output; ++i) {
			out_scales.push_back(output_attrs[i].scale);
			out_zps.push_back(output_attrs[i].zp);
		}

		post_process((int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
				box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

		// Draw results
		for (int i = 0; i < detect_result_group.count; i++) {
			detect_result_t* det = &(detect_result_group.results[i]);
			cv::rectangle(orig_img, cv::Point(det->box.left, det->box.top),
			cv::Point(det->box.right, det->box.bottom), cv::Scalar(0, 255, 0), 2);
			cv::putText(orig_img, det->name, cv::Point(det->box.left, det->box.top - 5),
						cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
		}


		// Write to ffmpeg
		cv::resize(orig_img, orig_img, cv::Size(1920, 1080));
		fwrite(orig_img.data, 1, orig_img.total() * orig_img.elemSize(), ffmpeg);

		rknn_outputs_release(ctx, io_num.n_output, outputs);
	}

	// Cleanup
	pclose(ffmpeg);
	cap.release();
	rknn_destroy(ctx);
	free(model_data);
	free(resize_buf);

	return 0;
}
