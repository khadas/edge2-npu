#include <jni.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <dirent.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <android/log.h>

#include "RgaUtils.h"
#include "im2d.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>
#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "lpd.h"
#include "lpr.h"
#include "lpc.h"
#include "rga.h"
#include "rknn_api.h"
#include "imgproc/imgproc_c.h"
#include "videoio.hpp"

using namespace std;

#define LOG_TAG "rknn_demo"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)

#define PERF_WITH_POST 1

/*-------------------------------------------
                  Functions
-------------------------------------------*/

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

//char*          lpd_model_name = NULL;
rknn_context   lpd_ctx;
int            lpd_width          = 0;
int            lpd_height         = 0;
int            lpd_channel        = 0;
std::vector<float> lpd_out_scales;
std::vector<int32_t> lpd_out_zps;
rknn_input_output_num lpd_io_num;
static unsigned char *lpd_model_data;

cv::Point2f dst_lpr[4] = {cv::Point2f(30, 1), cv::Point2f(250, 1), cv::Point2f(250, 31), cv::Point2f(30, 31)};
cv::Point2f dst_lpc[4] = {cv::Point2f(0, 0), cv::Point2f(32, 0), cv::Point2f(32, 32), cv::Point2f(0, 32)};
cv::Mat M;
cv::Mat imagedst;

//char*          lpr_model_name = NULL;
rknn_context   lpr_ctx;
int            lpr_width          = 0;
int            lpr_height         = 0;
int            lpr_channel        = 0;
std::vector<float> lpr_out_scales;
std::vector<int32_t> lpr_out_zps;
rknn_input_output_num lpr_io_num;
static unsigned char *lpr_model_data;

//char*          lpc_model_name = NULL;
rknn_context   lpc_ctx;
int            lpc_width          = 0;
int            lpc_height         = 0;
int            lpc_channel        = 0;
std::vector<float> lpc_out_scales;
std::vector<int32_t> lpc_out_zps;
rknn_input_output_num lpc_io_num;
static unsigned char *lpc_model_data;

rknn_input lpd_inputs[1];
rknn_input lpr_inputs[1];
rknn_input lpc_inputs[1];

rknn_output lpd_outputs[3];
rknn_output lpr_outputs[1];
rknn_output lpc_outputs[1];

const float    lpd_nms_threshold      = NMS_THRESH;
const float    lpd_box_conf_threshold = BOX_THRESH;
const float    lpr_conf_threshold     = STR_THRESH;
const float    lpc_conf_threshold     = COLOR_THRESH;

static char* jstringToChar(JNIEnv* env, jstring jstr) {
    char* rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("utf-8");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);

    if (alen > 0) {
        rtn = new char[alen + 1];
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}

int identify(int pic_width, int pic_height, int pic_channgel, int flip, unsigned char *pic_data, int pic_len, int *lpLen, float *scores, int *boxes, char *lpInfo)
{
    int orig_img_width  = pic_width;
    int orig_img_height = pic_height;

    int img_width;
    int img_height;

    cv::Mat orig_img = cv::imdecode(cv::Mat(1, pic_len, CV_8UC1, pic_data),cv::IMREAD_COLOR);
    cv::Mat img;
    if (!orig_img.data) {
        return -1;
    }

//    cv::imwrite("/data/user/0/com.wesion.demo/cache/orig_img1.jpg", orig_img);
//    cv::flip(orig_img, orig_img, flip);
//    cv::imwrite("/data/user/0/com.wesion.demo/cache/orig_img2.jpg", orig_img);

    if (orig_img_width >= orig_img_height)
    {
        img_width = orig_img_width;
        img_height = img_width;
        int x_padding = img_width - orig_img_width;
        int y_padding = img_height - orig_img_height;
        cv::copyMakeBorder(orig_img, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    else if (orig_img_width < orig_img_height)
    {
        img_height = orig_img_height;
        img_width = img_height;
        int x_padding = img_width - orig_img_width;
        int y_padding = img_height - orig_img_height;
        cv::copyMakeBorder(orig_img, img, 0, y_padding, 0, x_padding, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    cv::resize(img, img, cv::Size(lpd_width, lpd_height));
    cv::imwrite("/data/user/0/com.wesion.demo/cache/resize.jpg", img);

    float scale_w = (float)lpd_width / img_width;
    float scale_h = (float)lpd_height / img_height;

//    cv::copyMakeBorder(orig_img, img, 0, 840, 0, 0, cv::BorderTypes::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
//    cv::resize(img, img, cv::Size(lpd_width, lpd_height));
    detect_result_group_t lpd_detect_result_group;
    lpd_inference(&lpd_ctx, img, lpd_width, lpd_height, lpd_channel, lpd_box_conf_threshold, lpd_nms_threshold, img_width, img_height, lpd_io_num, lpd_inputs, lpd_outputs, &lpd_detect_result_group);

    char text[256];
    for (int i = 0; i < lpd_detect_result_group.count; i++) {
        detect_result_t* det_result = &(lpd_detect_result_group.results[i]);

//        int x1 = det_result->box.left;
//        int y1 = det_result->box.top;
//        int x2 = det_result->box.right;
//        int y2 = det_result->box.bottom;

        scores[i] = det_result->prop;
        boxes[i * 4 + 0] = det_result->box.left;
        boxes[i * 4 + 1] = det_result->box.top;
        boxes[i * 4 + 2] = det_result->box.right;
        boxes[i * 4 + 3] = det_result->box.bottom;

        cv::Point2f src[4];
        for (int j = 0; j < KEY_POINT_NUM; ++j)
        {
            if (det_result->point[j].conf < POINT_THRESH)
            {
                continue;
            }
            int ponit_x = det_result->point[j].x;
            int ponit_y = det_result->point[j].y;
            src[j] = cv::Point2f(ponit_x, ponit_y);
        }

        //lpr
        M = cv::getPerspectiveTransform(src, dst_lpr);
        cv::warpPerspective(orig_img, imagedst, M, cv::Size(280, 32));
        cv::cvtColor(imagedst, imagedst, cv::COLOR_BGR2GRAY);
        char* lpr_result[36];
        int len = 0;
        lpr_inference(&lpr_ctx, imagedst, lpr_io_num, lpr_inputs, lpr_outputs, lpr_conf_threshold, lpr_result, &len);

        //lpc
        M = cv::getPerspectiveTransform(src, dst_lpc);
        cv::warpPerspective(orig_img, imagedst, M, cv::Size(32, 32));
        char* lpc_result[1];
        lpc_inference(&lpc_ctx, imagedst, lpc_io_num, lpc_inputs, lpc_outputs, lpc_conf_threshold, lpc_result);

        std::string result;
        for (int ii = 0; ii < len; ++ii) {
            result = result + std::string(lpr_result[ii]);
        }

        lpLen[i] = (int)result.length();
        for (int jj = 0; jj <  lpLen[i]; jj++) {
            lpInfo[i * 36 + jj] = result.c_str()[jj];
        }

//        rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 3);
//        putText(orig_img, lpc_result[0], cv::Point(x1, y1 - 108), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
//        cv::imwrite("/data/user/0/com.wesion.demo/cache/out.jpg", orig_img);
    }
    return lpd_detect_result_group.count;
}


int init(char *lpd_model_name, char *lpd_label_name, char *lpr_model_name, char *lpr_label_name, char *lpc_model_name, char *lpc_label_name)
{
    create_lpd(lpd_model_name, lpd_label_name, &lpd_ctx, lpd_width, lpd_height, lpd_channel, lpd_out_scales, lpd_out_zps, lpd_io_num, lpd_model_data);
    create_lpr(lpr_model_name, lpr_label_name, &lpr_ctx, lpr_width, lpr_height, lpr_channel, lpr_out_scales, lpr_out_zps, lpr_io_num, lpr_model_data);
    create_lpc(lpc_model_name, lpc_label_name, &lpc_ctx, lpc_width, lpc_height, lpc_channel, lpc_out_scales, lpc_out_zps, lpc_io_num, lpc_model_data);

    memset(lpd_inputs, 0, sizeof(lpd_inputs));
    lpd_inputs[0].index        = 0;
    lpd_inputs[0].type         = RKNN_TENSOR_UINT8;
    lpd_inputs[0].size         = lpd_width * lpd_height * lpd_channel;
    lpd_inputs[0].fmt          = RKNN_TENSOR_NHWC;
    lpd_inputs[0].pass_through = 0;

    memset(lpd_outputs, 0, sizeof(lpd_outputs));
    for (int i = 0; i < lpd_io_num.n_output; i++) {
        lpd_outputs[i].want_float = 1;
    }

    memset(lpr_inputs, 0, sizeof(lpr_inputs));
    lpr_inputs[0].index        = 0;
    lpr_inputs[0].type         = RKNN_TENSOR_UINT8;
    lpr_inputs[0].size         = lpr_width * lpr_height * lpr_channel;
    lpr_inputs[0].fmt          = RKNN_TENSOR_NHWC;
    lpr_inputs[0].pass_through = 0;

    memset(lpr_outputs, 0, sizeof(lpr_outputs));
    for (int i = 0; i < lpr_io_num.n_output; i++) {
        lpr_outputs[i].want_float = 1;
    }

    memset(lpc_inputs, 0, sizeof(lpc_inputs));
    lpc_inputs[0].index        = 0;
    lpc_inputs[0].type         = RKNN_TENSOR_UINT8;
    lpc_inputs[0].size         = lpc_width * lpc_height * lpc_channel;
    lpc_inputs[0].fmt          = RKNN_TENSOR_NHWC;
    lpc_inputs[0].pass_through = 0;

    memset(lpc_outputs, 0, sizeof(lpc_outputs));
    for (int i = 0; i < lpc_io_num.n_output; i++) {
        lpc_outputs[i].want_float = 1;
    }

    return 0;
}

int deInit()
{
    release_lpd(&lpd_ctx, lpd_model_data);
    release_lpr(&lpr_ctx, lpr_model_data);
    release_lpc(&lpc_ctx, lpc_model_data);
    deinitPostProcess();

    return 0;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_Recognition_native_1identify(JNIEnv* env, jobject object, jint width, jint height,
                                                      jint channel,
                                                      jint flip,
                                                      jbyteArray data,
                                                      jintArray lpLen,
                                                      jfloatArray scores,
                                                      jintArray boxes,
                                                      jbyteArray lpInfo
) {
    int ret = 0;
    int len = env->GetArrayLength (data);
    jboolean outputCopy = JNI_FALSE;
    jint*  const i = env->GetIntArrayElements(lpLen, &outputCopy);
    jfloat* const s = env->GetFloatArrayElements(scores, &outputCopy);
    jint*  const b = env->GetIntArrayElements(boxes, &outputCopy);
    jbyte * const p = env->GetByteArrayElements(lpInfo, &outputCopy);

    unsigned char* buf = (unsigned char *)malloc(len);
    env->GetByteArrayRegion(data, 0, len, reinterpret_cast<jbyte*>(buf));
    ret = identify(width, height, channel, flip, buf, len, i, s, b, (char *)p);
    if (buf != nullptr) {
        free(buf);
    }

    env->ReleaseIntArrayElements(lpLen, i, 0);
    env->ReleaseFloatArrayElements(scores, s, 0);
    env->ReleaseIntArrayElements(boxes, b, 0);
    env->ReleaseByteArrayElements(lpInfo, p, 0);
    return ret;
}



extern "C" JNIEXPORT jint
Java_com_wesion_demo_Recognition_native_1init(JNIEnv* env, jobject object, jstring lpd_model_path, jstring lpd_label_path, jstring lpr_model_path, jstring lpr_label_path, jstring lpc_model_path, jstring lpc_label_path) {
    char *lpd_model_path_p = jstringToChar(env, lpd_model_path);
    char *lpd_label_path_p = jstringToChar(env, lpd_label_path);
    char *lpr_model_path_p = jstringToChar(env, lpr_model_path);
    char *lpr_label_path_p = jstringToChar(env, lpr_label_path);
    char *lpc_model_path_p = jstringToChar(env, lpc_model_path);
    char *lpc_label_path_p = jstringToChar(env, lpc_label_path);

    init(lpd_model_path_p, lpd_label_path_p, lpr_model_path_p, lpr_label_path_p, lpc_model_path_p, lpc_label_path_p);
    return 0;
}

extern "C" JNIEXPORT jint
Java_com_wesion_demo_Recognition_native_1deInit(JNIEnv* env, jobject object) {
    deInit();
    return 0;
}

