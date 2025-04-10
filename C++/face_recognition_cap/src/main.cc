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
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "retinaface.h"
#include "facenet.h"
#include "camera_util.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
#define WIDTH  1920
#define HEIGHT 1080

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  	char*          retinaface_model_name = NULL;
  	rknn_context   retinaface_ctx;
  	int            retinaface_width      = 0;
  	int            retinaface_height     = 0;
  	int            retinaface_channel    = 0;
  	std::vector<float> retinaface_out_scales;
  	std::vector<int32_t> retinaface_out_zps;
  	rknn_input_output_num retinaface_io_num;
  	static unsigned char *retinaface_model_data;
  	
  	float dst_landmark[5][2] = {{54.7065, 73.8519},
				    {105.0454, 73.5734},
				    {80.036, 102.4808},
				    {59.3561, 131.9507},
				    {89.6141, 131.7201}};
	cv::Mat dst(5, 2, CV_32FC1, dst_landmark);
	memcpy(dst.data, dst_landmark, 2 * 5 * sizeof(float));
  	
  	char*          facenet_model_name = NULL;
  	rknn_context   facenet_ctx;
  	int            facenet_width      = 0;
  	int            facenet_height     = 0;
  	int            facenet_channel    = 0;
  	rknn_input_output_num facenet_io_num;
  	static unsigned char *facenet_model_data;
  	float *facenet_result;

  	const float    nms_threshold      = NMS_THRESH;
  	const float    box_conf_threshold = BOX_THRESH;
  	const float    facenet_threshold  = FACENET_THRESH;
  	struct timeval start_time, stop_time;
  	int            ret;

  	if (argc != 5) {
		printf("Usage: %s <retinaface model> <facenet model> <usb or mipi> <device number> \n", argv[0]);
		return -1;
  	}

  	printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);

  	retinaface_model_name = (char*)argv[1];
  	facenet_model_name = (char*)argv[2];
  	std::string camera_type = argv[3];
  	std::string device_number = argv[4];
  	
  	create_retinaface(retinaface_model_name, &retinaface_ctx, retinaface_width, retinaface_height, retinaface_channel, retinaface_out_scales, retinaface_out_zps, retinaface_io_num, retinaface_model_data);
  	create_facenet(facenet_model_name, &facenet_ctx, facenet_width, facenet_height, facenet_channel, facenet_io_num, facenet_model_data);
  	
  	rknn_input retinaface_inputs[1];
  	memset(retinaface_inputs, 0, sizeof(retinaface_inputs));
  	retinaface_inputs[0].index        = 0;
  	retinaface_inputs[0].type         = RKNN_TENSOR_UINT8;
  	retinaface_inputs[0].size         = retinaface_width * retinaface_height * retinaface_channel;
  	retinaface_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	retinaface_inputs[0].pass_through = 0;
  	
  	rknn_output retinaface_outputs[retinaface_io_num.n_output];
  	memset(retinaface_outputs, 0, sizeof(retinaface_outputs));
  	for (int i = 0; i < retinaface_io_num.n_output; i++) {
  		if (i != 1)
  		{
			retinaface_outputs[i].want_float = 0;
		}
		else
		{
			retinaface_outputs[i].want_float = 1;
		}
  	}
  	
  	rknn_input facenet_inputs[1];
  	memset(facenet_inputs, 0, sizeof(facenet_inputs));
  	facenet_inputs[0].index        = 0;
  	facenet_inputs[0].type         = RKNN_TENSOR_UINT8;
  	facenet_inputs[0].size         = facenet_width * facenet_height * facenet_channel;
  	facenet_inputs[0].fmt          = RKNN_TENSOR_NHWC;
  	facenet_inputs[0].pass_through = 0;
  	
  	rknn_output facenet_outputs[facenet_io_num.n_output];
  	memset(facenet_outputs, 0, sizeof(facenet_outputs));
  	for (int i = 0; i < facenet_io_num.n_output; i++) {
		facenet_outputs[i].want_float = 1;
  	}
  	
  	cv::namedWindow("Image Window");
  	cv::Mat orig_img;
	cv::Mat img;
	
	if (camera_type == "usb") {
		ret = load_usb_camera(device_number, WIDTH, HEIGHT);
	}
	else if (camera_type == "mipi") {
		ret = load_mipi_camera(device_number, WIDTH, HEIGHT);
	}
	else {
		std::cout << "Unsupport camera type : " << camera_type << " !!!" << std::endl;
	}
  	
  	float scale_w, scale_h;
	int resize_w, resize_h, padding;
	if (WIDTH > HEIGHT) {
		scale_w = (float)retinaface_width / WIDTH;
		scale_h = scale_w;
		resize_w = retinaface_width;
		resize_h = (int)(resize_w * HEIGHT / WIDTH);
		padding = resize_w - resize_h;
	}
	else {
		scale_h = (float)retinaface_height / HEIGHT;
		scale_w = scale_h;
		resize_h = retinaface_height;
		resize_w = (int)(resize_h * WIDTH / HEIGHT);
		padding = resize_h - resize_w;
	}

  	std::vector<float>    out_scales;
  	std::vector<int32_t>  out_zps;
  	char text[256];

	int x1,y1,x2,y2;
	
	std::string face_lib = "./data/face_feature_lib/";
	DIR *pDir;
	struct dirent *ptr;
	if (!(pDir = opendir(face_lib.c_str())))
	{
		printf("Feature library doesn't Exist!\n");
		return -1;
	}
	
	std::vector<float*> lib_feature;
	std::vector<std::string> lib_face_name;
	while ((ptr = readdir(pDir)) != 0)
	{
		if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
		{
			std::ifstream infile(face_lib + ptr->d_name);
			std::string tmp;
			
			float* tmp_lib_feature = new float[128];
			
			int i = 0;
			while (getline(infile, tmp))
			{
				tmp_lib_feature[i] = atof(tmp.c_str());
				i++;
			}
			infile.close();
			
			lib_face_name.push_back(((std::string)ptr->d_name).substr(0, ((std::string)ptr->d_name).find_last_of(".")));
			lib_feature.push_back(tmp_lib_feature);
		}
	}
	
	float total_time = 0;
	int n = 0;
	
  	while(1){
		gettimeofday(&start_time, NULL);
		if (camera_type == "usb") {
			read_usb_frame(&orig_img);
		}
		else if (camera_type == "mipi") {
			read_mipi_frame(&orig_img);
		}
		
		cv::resize(orig_img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
		detect_result_group_t retinaface_detect_result_group;
		if (WIDTH > HEIGHT) {
			cv::copyMakeBorder(img, img, 0, padding, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, WIDTH, WIDTH, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
		}
		else{
			cv::copyMakeBorder(img, img, 0, 0, 0, padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			retinaface_inference(&retinaface_ctx, img, retinaface_width, retinaface_height, retinaface_channel, box_conf_threshold, nms_threshold, HEIGHT, HEIGHT, retinaface_io_num, retinaface_inputs, retinaface_outputs, retinaface_out_scales, retinaface_out_zps, &retinaface_detect_result_group);
		}
  		
  		for (int i = 0; i < retinaface_detect_result_group.count; i++) {
			float landmark[5][2] = {{(float)retinaface_detect_result_group.results[i].point.point_1_x, (float)retinaface_detect_result_group.results[i].point.point_1_y},
						{(float)retinaface_detect_result_group.results[i].point.point_2_x, (float)retinaface_detect_result_group.results[i].point.point_2_y},
						{(float)retinaface_detect_result_group.results[i].point.point_3_x, (float)retinaface_detect_result_group.results[i].point.point_3_y},
						{(float)retinaface_detect_result_group.results[i].point.point_4_x, (float)retinaface_detect_result_group.results[i].point.point_4_y},
						{(float)retinaface_detect_result_group.results[i].point.point_5_x, (float)retinaface_detect_result_group.results[i].point.point_5_y}};
			
			cv::Mat src(5, 2, CV_32FC1, landmark);
			memcpy(src.data, landmark, 2 * 5 * sizeof(float));
				
			cv::Mat M = similarTransform(src, dst);
			cv::Mat warp;
			cv::warpPerspective(orig_img, warp, M, cv::Size(facenet_width, facenet_height));
			cv::cvtColor(warp, warp, cv::COLOR_BGR2RGB);
			
			facenet_inference(&facenet_ctx, warp, facenet_io_num, facenet_inputs, facenet_outputs, &facenet_result);
			
			float max_score = 0;
			std::string name = "stranger";
			for (int i = 0; i < lib_feature.size(); i++)
  			{
				float cos_similar;
				
				cos_similar = cos_similarity(facenet_result, lib_feature[i]);
				if (cos_similar >= facenet_threshold && cos_similar > max_score)
				{
					max_score = cos_similar;
					name = lib_face_name[i];
				}
  			}
  			facenet_output_release(&facenet_ctx, facenet_io_num, facenet_outputs);
  			
  			int x1 = retinaface_detect_result_group.results[i].box.left;
			int y1 = retinaface_detect_result_group.results[i].box.top;
			int x2 = retinaface_detect_result_group.results[i].box.right;
			int y2 = retinaface_detect_result_group.results[i].box.bottom;
			
			rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 1);
			putText(orig_img, name, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
  		}
  		cv::imshow("Image Window",orig_img);
		cv::waitKey(1);
		
		gettimeofday(&stop_time, NULL);
		//printf("total run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
		total_time += (__get_us(stop_time) - __get_us(start_time)) / 1000;
		n++;
		if (n == 10)
		{
			printf("average time : %f ms\n", (total_time / 10));
			total_time = 0;
			n = 0;
		}
  	}
  	if (camera_type == "usb") {
		close_usb_camera();
	}
	else if (camera_type == "mipi") {
		close_mipi_camera();
	}

	release_retinaface(&retinaface_ctx, retinaface_model_data);
	release_facenet(&facenet_ctx, facenet_model_data);
  	return 0;
}
