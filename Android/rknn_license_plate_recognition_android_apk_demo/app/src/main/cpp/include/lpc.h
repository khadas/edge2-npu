#ifndef __LPC_H__
#define __LPC_H__

#include <stdint.h>
#include <vector>
#include "rknn_api.h"

int create_lpc(char *model_name, char *label_name, rknn_context *ctx, int &width, int &height, int &channel, std::vector<float> &out_scales, std::vector<int32_t> &out_zps, rknn_input_output_num &io_num, unsigned char *model_data);

int lpc_inference(rknn_context *ctx, cv::Mat img, rknn_input_output_num io_num, rknn_input *inputs, rknn_output *outputs, float conf_threshold, char* result[]);

void release_lpc(rknn_context *ctx, unsigned char *model_data);
#endif //__LPC_H__
