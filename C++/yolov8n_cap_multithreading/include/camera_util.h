#ifndef _CAMERA_UTIL_H_

#include <string.h>
#include <opencv2/opencv.hpp>

#define REQ_COUNT 4

struct Buffer {
    void* start;
    size_t length;
};

int load_usb_camera(std::string device, int camera_width, int camera_height);
void read_usb_frame(cv::Mat *orig_img);
void close_usb_camera();
int load_mipi_camera(std::string device, int camera_width, int camera_height);
void read_mipi_frame(cv::Mat *orig_img);
void close_mipi_camera();

#endif
