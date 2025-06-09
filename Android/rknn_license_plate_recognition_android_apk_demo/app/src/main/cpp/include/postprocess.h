#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM     1
#define KEY_POINT_NUM     4
#define STR_CLASS_NUM     65
#define COLOR_CLASS_NUM   2
#define NMS_THRESH        0.6
#define BOX_THRESH        0.5
#define POINT_THRESH      0.3
#define STR_THRESH        0.3
#define COLOR_THRESH      0.5
#define PROP_BOX_SIZE     (64+OBJ_CLASS_NUM+KEY_POINT_NUM*3)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct _KEY_POINT
{
    int x;
    int y;
    float conf;
} KEY_POINT;

typedef struct __detect_result_t
{
    char* name;
    BOX_RECT box;
    KEY_POINT point[KEY_POINT_NUM];
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int load_lpd_label(char *label_name);

int lpd_post_process(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 detect_result_group_t *group);

int load_lpr_label(char *label_name);

int lpr_post_process(float *input0, float conf_threshold, char *result[], int *len);

int load_lpc_label(char *label_name);

int lpc_post_process(float *input0, float conf_threshold, char *result[]);

void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
