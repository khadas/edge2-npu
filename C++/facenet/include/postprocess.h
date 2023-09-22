#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     89
#define BOX_THRESH        0.25

void l2_normalize(float* input);

float compare_eu_distance(float* input1, float* input2);

float cos_similarity(float* input1, float* input2);

#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
