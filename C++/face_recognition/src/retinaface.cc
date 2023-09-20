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

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "retinaface.h"
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
int create_retinaface(char *model_name, rknn_context *ctx, int &width, int &height, int &channel, std::vector<float> &out_scales, std::vector<int32_t> &out_zps, rknn_input_output_num &io_num, unsigned char *model_data)
{
  	int            status     = 0;
  	int            ret;
  	
  	/* Create the neural network */
  	printf("Loading retinaface model...\n");
  	int model_data_size = 0;
  	model_data          = load_model(model_name, &model_data_size);
  	ret                 = rknn_init(ctx, model_data, model_data_size, 0, NULL);
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	
  	rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  	ret = rknn_set_core_mask(*ctx, core_mask);
  	if (ret < 0) {
		printf("rknn_set_core_mask error ret=%d\n", ret);
		return -1;
  	}

  	rknn_sdk_version version;
  	ret = rknn_query(*ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

  	ret = rknn_query(*ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  	rknn_tensor_attr input_attrs[io_num.n_input];
  	memset(input_attrs, 0, sizeof(input_attrs));
  	for (int i = 0; i < io_num.n_input; i++) {
		input_attrs[i].index = i;
		ret                  = rknn_query(*ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
		if (ret < 0) {
	  		printf("rknn_init error ret=%d\n", ret);
	  		return -1;
		}
		dump_tensor_attr(&(input_attrs[i]));
  	}
  	
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

  	rknn_tensor_attr output_attrs[io_num.n_output];
  	memset(output_attrs, 0, sizeof(output_attrs));
  	for (int i = 0; i < io_num.n_output; i++) {
		output_attrs[i].index = i;
		ret                   = rknn_query(*ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
		dump_tensor_attr(&(output_attrs[i]));
  	}
  	
  	for (int i = 0; i < io_num.n_output; ++i) {
		out_scales.push_back(output_attrs[i].scale);
		out_zps.push_back(output_attrs[i].zp);
  	}
  	return ret;
}

int retinaface_inference(rknn_context *ctx, cv::Mat img, int width, int height, int channel, float box_conf_threshold, float nms_threshold, int img_width, int img_height, rknn_input_output_num io_num, rknn_input *inputs, rknn_output *outputs, std::vector<float> out_scales, std::vector<int32_t> out_zps, detect_result_group_t *detect_result_group)
{
  	// You may not need resize when src resulotion equals to dst resulotion
  	inputs[0].buf = (void*)img.data;
  	
  	int ret;

  	ret = rknn_inputs_set(*ctx, io_num.n_input, inputs);

  	ret = rknn_run(*ctx, NULL);
  	ret = rknn_outputs_get(*ctx, io_num.n_output, outputs, NULL);

  	// post process
  	float scale_w = (float)width / img_width;
  	float scale_h = (float)height / img_height;
  	
  	
  	memset(detect_result_group, 0, sizeof(detect_result_group_t));

  	post_process((int8_t*)outputs[0].buf, (float*)outputs[1].buf, (int8_t*)outputs[2].buf, height, width,
			box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, detect_result_group);
  	
  	ret = rknn_outputs_release(*ctx, io_num.n_output, outputs);
  	return ret;
}

void release_retinaface(rknn_context *ctx, unsigned char *model_data)
{
  	deinitPostProcess();

  	// release
  	int ret;
  	ret = rknn_destroy(*ctx);
  	
  	if (model_data) {
		free(model_data);
  	}
}
