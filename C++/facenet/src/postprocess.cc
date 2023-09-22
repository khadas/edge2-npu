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

void l2_normalize(float* input)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input[i] * input[i];
	}
	sum = sqrt(sum);
	for (int i = 0; i < 128; ++i)
	{
		input[i] = input[i] / sum;
	}
}

float eu_distance(float* input)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input[i] * input[i];
	}
	sum = sqrt(sum);
	return sum;
}

float compare_eu_distance(float* input1, float* input2)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + (input1[i] - input2[i]) * (input1[i] - input2[i]);
	}
	sum = sqrt(sum);
	return sum;
}

float cos_similarity(float* input1, float* input2)
{
	float sum = 0;
	for (int i = 0; i < 128; ++i)
	{
		sum = sum + input1[i] * input2[i];
	}
	float tmp1 = eu_distance(input1);
	float tmp2 = eu_distance(input2);
	return sum / (tmp1 * tmp2);
}
