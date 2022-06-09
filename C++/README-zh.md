## RKNPU For Edge2 / C++ Demo

### Docs

关于API接口如何使用，请参考doc文件夹下的文档。

### 编译demo

以yolov5为例，其他demo类似

```sh
$ cd yolov5
$ ./build.sh
```

编译会生成install文件夹。


### 运行demo

以yolov5为例，其他demo类似。

```sh
$ cd install/yolov5
$ ./yolov5 data/model/yolov5s-640-640.rknn data/img/bus.jpg 
post process config: box_conf_threshold = 0.50, nms_threshold = 0.60
Read data/img/bus.jpg ...
img width = 640, img height = 640
Loading mode...
sdk version: 1.3.0 (c193be371@2022-05-04T20:16:33) driver version: 0.7.2
model input num: 1, output num: 3
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
  index=0, name=output, n_dims=5, dims=[1, 3, 85, 80], n_elems=1632000, size=1632000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=77, scale=0.080445
  index=1, name=371, n_dims=5, dims=[1, 3, 85, 40], n_elems=408000, size=408000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=56, scale=0.080794
  index=2, name=390, n_dims=5, dims=[1, 3, 85, 20], n_elems=102000, size=102000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=69, scale=0.081305
model is NHWC input fmt
model input height=640, width=640, channel=3
once run use 32.872000 ms
loadLabelName ./data/coco_80_labels_list.txt
person @ (474 250 559 523) 0.996784
person @ (112 238 208 521) 0.992214
bus @ (99 141 557 445) 0.976798
person @ (211 242 285 509) 0.976798
loop count = 10 , average run  26.577900 ms
```

