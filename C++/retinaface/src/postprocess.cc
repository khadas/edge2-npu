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

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>

static char* labels[OBJ_CLASS_NUM];

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
  float i = w * h;
  float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
  return u <= 0.f ? 0.f : (i / u);
}

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

int post_process(int8_t* input0, float* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> keypoint;
  std::vector<float> objProbs;
  
  float min_sizes[3][2] = {{16, 32}, {64, 128}, {256, 512}};
  float variance[2] = {0.1, 0.2};
  int grid[3] = {80, 40, 20};

  int validCount = 0;
  int initial = 0;
  int index;
  float conf;
  
  for (int n = 0; n < 3; ++n)
  {
  	if (n == 1)
  	{
  		initial = pow(grid[0], 2) * 2;
  	}
  	else if (n == 2)
  	{
  		initial = (pow(grid[0], 2) + pow(grid[1], 2)) * 2;
  	}
  	for (int i = 0; i < grid[n]; ++i)
  	{
  		for (int j = 0; j < grid[n]; ++j)
  		{
  			for (int k = 0; k < 2; ++k)
  			{
  				index = initial + (i * grid[n] + j) * 2 + k;
  				conf = input1[index * 2 + 1];
  				if (conf >= conf_threshold)
  				{
  					float x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input0[index * 4 + 0], qnt_zps[0], qnt_scales[0]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input0[index * 4 + 1], qnt_zps[0], qnt_scales[0]) * variance[0] * min_sizes[n][k] / model_in_h;
  					float w = min_sizes[n][k] / model_in_w * expf(deqnt_affine_to_f32(input0[index * 4 + 2], qnt_zps[0], qnt_scales[0]) * variance[1]);
  					float h = min_sizes[n][k] / model_in_h * expf(deqnt_affine_to_f32(input0[index * 4 + 3], qnt_zps[0], qnt_scales[0]) * variance[1]);
  					
  					float point_1_x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 0], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float point_1_y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 1], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_h;
  					float point_2_x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 2], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float point_2_y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 3], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_h;
  					float point_3_x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 4], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float point_3_y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 5], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_h;
  					float point_4_x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 6], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float point_4_y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 7], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_h;
  					float point_5_x = (j + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 8], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_w;
  					float point_5_y = (i + 0.5) / grid[n] + deqnt_affine_to_f32(input2[index * 10 + 9], qnt_zps[2], qnt_scales[2]) * variance[0] * min_sizes[n][k] / model_in_h;
  					
  					filterBoxes.push_back(x);
  					filterBoxes.push_back(y);
  					filterBoxes.push_back(w);
  					filterBoxes.push_back(h);
  					
  					keypoint.push_back(point_1_x);
  					keypoint.push_back(point_1_y);
  					keypoint.push_back(point_2_x);
  					keypoint.push_back(point_2_y);
  					keypoint.push_back(point_3_x);
  					keypoint.push_back(point_3_y);
  					keypoint.push_back(point_4_x);
  					keypoint.push_back(point_4_y);
  					keypoint.push_back(point_5_x);
  					keypoint.push_back(point_5_y);
  					
  					objProbs.push_back(conf);
  					
  					validCount++;
  				}
  			}
  		}
  	}
  }
  
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  for (int i = 0; i < validCount; ++i)
  {
    if (objProbs[i] != -1)
    {
    	float x0min = filterBoxes[i * 4 + 0] - filterBoxes[i * 4 + 2] / 2;
    	float y0min = filterBoxes[i * 4 + 1] - filterBoxes[i * 4 + 3] / 2;
    	float x0max = filterBoxes[i * 4 + 0] + filterBoxes[i * 4 + 2] / 2;
    	float y0max = filterBoxes[i * 4 + 1] + filterBoxes[i * 4 + 3] / 2;
    	for (int j = i + 1; j < validCount; ++j)
    	{
    		if (objProbs[j] != -1)
    		{
    			float x1min = filterBoxes[j * 4 + 0] - filterBoxes[j * 4 + 2] / 2;
		    	float y1min = filterBoxes[j * 4 + 1] - filterBoxes[j * 4 + 3] / 2;
		    	float x1max = filterBoxes[j * 4 + 0] + filterBoxes[j * 4 + 2] / 2;
		    	float y1max = filterBoxes[j * 4 + 1] + filterBoxes[j * 4 + 3] / 2;
		    	
		    	float iou = CalculateOverlap(x0min, y0min, x0max, y0max, x1min, y1min, x1max, y1max);
		    	
		    	if (iou > nms_threshold)
		    	{
		    		if (objProbs[i] >= objProbs[j])
		    		{
		    			objProbs[j] = -1;
		    		}
		    		else
		    		{
		    			objProbs[i] = -1;
		    			break;
		    		}
		    	}
    		}
    	}
    }
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (objProbs[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }

    float x1       = (filterBoxes[i * 4 + 0] - filterBoxes[i * 4 + 2] / 2) * model_in_w;
    float y1       = (filterBoxes[i * 4 + 1] - filterBoxes[i * 4 + 3] / 2) * model_in_h;
    float x2       = (filterBoxes[i * 4 + 0] + filterBoxes[i * 4 + 2] / 2) * model_in_w;
    float y2       = (filterBoxes[i * 4 + 1] + filterBoxes[i * 4 + 3] / 2) * model_in_h;
    float obj_conf = objProbs[i];
    float point_1_x = keypoint[i * 10 + 0] * model_in_w;
    float point_1_y = keypoint[i * 10 + 1] * model_in_h;
    float point_2_x = keypoint[i * 10 + 2] * model_in_w;
    float point_2_y = keypoint[i * 10 + 3] * model_in_h;
    float point_3_x = keypoint[i * 10 + 4] * model_in_w;
    float point_3_y = keypoint[i * 10 + 5] * model_in_h;
    float point_4_x = keypoint[i * 10 + 6] * model_in_w;
    float point_4_y = keypoint[i * 10 + 7] * model_in_h;
    float point_5_x = keypoint[i * 10 + 8] * model_in_w;
    float point_5_y = keypoint[i * 10 + 9] * model_in_h;

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = "face";
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);
    group->results[last_count].point.point_1_x = (int)(clamp(point_1_x, 0, model_in_w) / scale_w);
    group->results[last_count].point.point_1_y = (int)(clamp(point_1_y, 0, model_in_h) / scale_h);
    group->results[last_count].point.point_2_x = (int)(clamp(point_2_x, 0, model_in_w) / scale_w);
    group->results[last_count].point.point_2_y = (int)(clamp(point_2_y, 0, model_in_h) / scale_h);
    group->results[last_count].point.point_3_x = (int)(clamp(point_3_x, 0, model_in_w) / scale_w);
    group->results[last_count].point.point_3_y = (int)(clamp(point_3_y, 0, model_in_h) / scale_h);
    group->results[last_count].point.point_4_x = (int)(clamp(point_4_x, 0, model_in_w) / scale_w);
    group->results[last_count].point.point_4_y = (int)(clamp(point_4_y, 0, model_in_h) / scale_h);
    group->results[last_count].point.point_5_x = (int)(clamp(point_5_x, 0, model_in_w) / scale_w);
    group->results[last_count].point.point_5_y = (int)(clamp(point_5_y, 0, model_in_h) / scale_h);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}
