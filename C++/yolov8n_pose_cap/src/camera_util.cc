#include <string.h>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include "camera_util.h"

#define CHECK_IOCTL(fd, request, arg) \
    if (ioctl(fd, request, arg) == -1) { \
        std::cerr << "IOCTL failed: " #request << std::endl; \
        exit(EXIT_FAILURE); \
    }


int fd;
v4l2_buffer buf;
Buffer *buffers = new Buffer[REQ_COUNT];
v4l2_format fmt = {};
int width, height;

int load_usb_camera(std::string device, int camera_width, int camera_height)
{
	std::string prefix = "/dev/video";
    fd = open((prefix + device).c_str(), O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return EXIT_FAILURE;
    }

    v4l2_capability cap;
    CHECK_IOCTL(fd, VIDIOC_QUERYCAP, &cap);
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        std::cerr << "Device does not support video capture" << std::endl;
        close(fd);
        return EXIT_FAILURE;
    }

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = camera_width;
    fmt.fmt.pix.height = camera_height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    CHECK_IOCTL(fd, VIDIOC_S_FMT, &fmt);

    v4l2_requestbuffers req = {};
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    req.count = REQ_COUNT;
    CHECK_IOCTL(fd, VIDIOC_REQBUFS, &req);

    for (unsigned i = 0; i < req.count; ++i) {
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        CHECK_IOCTL(fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("Memory mapping failed");
            exit(EXIT_FAILURE);
        }
        CHECK_IOCTL(fd, VIDIOC_QBUF, &buf);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CHECK_IOCTL(fd, VIDIOC_STREAMON, &type);

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

    return EXIT_SUCCESS;
}

void read_usb_frame(cv::Mat *orig_img)
{
    CHECK_IOCTL(fd, VIDIOC_DQBUF, &buf);
    cv::Mat raw_data(1, buf.bytesused, CV_8UC1, buffers[buf.index].start);
    *orig_img = cv::imdecode(raw_data, cv::IMREAD_COLOR);
    CHECK_IOCTL(fd, VIDIOC_QBUF, &buf);
}

void close_usb_camera()
{
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CHECK_IOCTL(fd, VIDIOC_STREAMOFF, &type);
    for (unsigned i = 0; i < REQ_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    delete[] buffers;
    close(fd);
}

int load_mipi_camera(std::string device, int camera_width, int camera_height)
{
	std::string prefix = "/dev/video";
    fd = open((prefix + device).c_str(), O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return EXIT_FAILURE;
    }

    v4l2_capability cap;
    CHECK_IOCTL(fd, VIDIOC_QUERYCAP, &cap);
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
        std::cerr << "Device does not support video capture" << std::endl;
        close(fd);
        return EXIT_FAILURE;
    }

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = camera_width;
    fmt.fmt.pix_mp.height = camera_height;
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt.fmt.pix_mp.num_planes = 2;
    CHECK_IOCTL(fd, VIDIOC_S_FMT, &fmt);
    
    width = fmt.fmt.pix_mp.width;
    height = fmt.fmt.pix_mp.height;

    v4l2_requestbuffers req = {};
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    req.count = REQ_COUNT;
    CHECK_IOCTL(fd, VIDIOC_REQBUFS, &req);

    for (unsigned i = 0; i < req.count; ++i) {
        v4l2_plane planes[VIDEO_MAX_PLANES];
        
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = VIDEO_MAX_PLANES;
        CHECK_IOCTL(fd, VIDIOC_QUERYBUF, &buf);

        buffers[i].length = buf.m.planes[0].length;
        buffers[i].start = mmap(NULL, buf.m.planes[0].length,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED, fd, buf.m.planes[0].m.mem_offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("Memory mapping failed");
            exit(EXIT_FAILURE);
        }
        CHECK_IOCTL(fd, VIDIOC_QBUF, &buf);
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    CHECK_IOCTL(fd, VIDIOC_STREAMON, &type);

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	buf.memory = V4L2_MEMORY_MMAP;

    return EXIT_SUCCESS;
}

void read_mipi_frame(cv::Mat *orig_img)
{
    CHECK_IOCTL(fd, VIDIOC_DQBUF, &buf);
    cv::Mat raw_data(height * 3 / 2, width, CV_8UC1, buffers[buf.index].start);
    
    cv::cvtColor(raw_data, *orig_img, cv::COLOR_YUV2BGR_NV12);
    CHECK_IOCTL(fd, VIDIOC_QBUF, &buf);
}


void close_mipi_camera()
{
	v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CHECK_IOCTL(fd, VIDIOC_STREAMOFF, &type);
    for (unsigned i = 0; i < REQ_COUNT; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    delete[] buffers;
    close(fd);
}
