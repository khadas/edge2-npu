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
#include <dirent.h>
#include <iostream>
#include <fstream>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
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
		printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
		return -1;
  	}

  	model_name       = (char*)argv[1];
  	char* image_name = argv[2];
  	cv::Mat orig_img, img;

  	/* Create the neural network */
  	printf("Loading mode...\n");
  	int            model_data_size = 0;
  	unsigned char* model_data      = load_model(model_name, &model_data_size);
  	ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  	if (ret < 0) {
		printf("rknn_init error ret=%d\n", ret);
		return -1;
  	}
  	
  	rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  	ret = rknn_set_core_mask(ctx, core_mask);
  	if (ret < 0) {
		printf("rknn_set_core_mask error ret=%d\n", ret);
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
		width   = input_attrs[0].dims[3];
		height  = input_attrs[0].dims[2];
  	} else {
		printf("model is NHWC input fmt\n");
		width   = input_attrs[0].dims[2];
		height  = input_attrs[0].dims[1];
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

	rknn_output outputs[io_num.n_output];
  	memset(outputs, 0, sizeof(outputs));
  				
  	for (int i = 0; i < io_num.n_output; i++) {
		outputs[i].want_float = 1;
  	}
  	
  	float* result;
  	
  	gettimeofday(&start_time, NULL);
  	if (!(atoi(image_name) == 1))
  	{
  		std::string face_lib = "../../face_feature_lib/";
  		DIR *pDir;
  		struct dirent *ptr;
  		if (!(pDir = opendir(face_lib.c_str())))
  		{
  			printf("Feature library doesn't Exist!\n");
  		}
  		orig_img = cv::imread(image_name);
  		if (!orig_img.data) {
			printf("cv::imread %s fail!\n", image_name);
			return -1;
  		}
  		cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  		
  		if (orig_img.cols >= orig_img.rows)
  		{
  			int new_height = (int)(orig_img.rows * 160 / orig_img.cols);
  			cv::resize(img, img, cv::Size(160, new_height));
  			int y_padding = 160 - new_height;
  			cv::copyMakeBorder(img, img, 0, y_padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		}
  		else
  		{
  			int new_width = (int)(orig_img.cols * 160 / orig_img.rows);
  			cv::resize(img, img, cv::Size(new_width, 160));
  			int x_padding = 160 - new_width;
  			cv::copyMakeBorder(img, img, 0, 0, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  		}
  		
  		inputs[0].buf = (void*)img.data;
  		rknn_inputs_set(ctx, io_num.n_input, inputs);
  		
  		ret = rknn_run(ctx, NULL);
  		ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  		result = (float*)outputs[0].buf;
  				
  		l2_normalize(result);
  		
  		while ((ptr = readdir(pDir)) != 0)
  		{
  			if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  			{
  				std::ifstream infile(face_lib + ptr->d_name);
  				std::cout << "\n" << ptr->d_name << std::endl;
  				std::string tmp;
  				float lib_feature[128];
  				float tmp2, tmp3;
  				
  				int i = 0;
  				while (getline(infile, tmp))
  				{
  					lib_feature[i] = atof(tmp.c_str());
  					i++;
  				}
  				infile.close();
  				
  				tmp2 = compare_eu_distance(result, lib_feature);
  				tmp3 = cos_similarity(result, lib_feature);
  				printf("eu_distance:%f\n", tmp2);
  				printf("cos_similarity:%f\n", tmp3);
  			}
  		}
  	}
  	else
  	{
  		std::string save_path = "../../face_feature_lib/";
  		std::string path = "./data/img/";
  		DIR *pDir;
  		struct dirent *ptr;
  		if (!(pDir = opendir(save_path.c_str())))
  		{
  			system(("mkdir " + save_path).c_str());
  		}
  		if (!(pDir = opendir(path.c_str())))
  		{
  			printf("Face path doesn't Exist!\n");
  		}
  		while ((ptr = readdir(pDir)) != 0)
  		{
  			if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
  			{
  				orig_img = cv::imread(path + ptr->d_name);
  				cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  				
  				if (orig_img.cols >= orig_img.rows)
		  		{
		  			int new_height = (int)(orig_img.rows * 160 / orig_img.cols);
		  			cv::resize(img, img, cv::Size(160, new_height));
		  			int y_padding = 160 - new_height;
		  			cv::copyMakeBorder(img, img, 0, y_padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		  		}
		  		else
		  		{
		  			int new_width = (int)(orig_img.cols * 160 / orig_img.rows);
		  			cv::resize(img, img, cv::Size(new_width, 160));
		  			int x_padding = 160 - new_width;
		  			cv::copyMakeBorder(img, img, 0, 0, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		  		}
  				
  				inputs[0].buf = (void*)img.data;
  				rknn_inputs_set(ctx, io_num.n_input, inputs);
  				  				
  				ret = rknn_run(ctx, NULL);
  				ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  				result = (float*)outputs[0].buf;
  				
  				l2_normalize(result);
  				
  				std::ofstream outfile(save_path + ((std::string)ptr->d_name).substr(0, ((std::string)ptr->d_name).find_last_of(".")) + ".dat", std::ofstream::out);
  				for (int i = 0; i < 128; ++i)
  				{
  					outfile << result[i] << "\n";
  				}
  				outfile.close();
  			}
  		}
  	}
  	
  	gettimeofday(&stop_time, NULL);
  	printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

  	ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  	// release
  	ret = rknn_destroy(ctx);

  	if (model_data) {
		free(model_data);
  	}

  	return 0;
}
