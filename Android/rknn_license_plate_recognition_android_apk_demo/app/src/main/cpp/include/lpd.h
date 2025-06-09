#ifndef __LPD_H__
#define __LPD_H__

#include <stdint.h>
#include <vector>
#include "rknn_api.h"

int create_lpd(char *model_name, char *label_name, rknn_context *ctx, int &width, int &height, int &channel, std::vector<float> &out_scales, std::vector<int32_t> &out_zps, rknn_input_output_num &io_num, unsigned char *model_data);

int lpd_inference(rknn_context *ctx, cv::Mat img, int width, int height, int channel, float box_conf_threshold, float nms_threshold, int img_width, int img_height, rknn_input_output_num io_num, rknn_input *inputs, rknn_output *outputs, detect_result_group_t *detect_result_group);

void release_lpd(rknn_context *ctx, unsigned char *model_data);
#endif //__LPD_H__
