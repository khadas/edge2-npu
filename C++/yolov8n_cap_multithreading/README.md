## Compile demo

```sh
$ bash build.sh
```

## Inference demo

```sh
$ cd install/yolov8n_cap_multithreading
$ ./yolov8n_cap_multithreading <rknn model> <camera type> <device num> <thread num>

# USB Camera
$ ./yolov8n_cap_multithreading data/model/yolov8n.rknn usb 60 2

# MIPI Camera
$ ./yolov8n_cap_multithreading data/model/yolov8n.rknn mipi 42 2
```

