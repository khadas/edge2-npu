## Compile demo

```sh
$ bash build.sh
```

## Inference demo

This demo integrates retinaface and facenet.

Before inference, please run face_recognition to generate face_feature_lib and copy the library here.

```sh
# Identification
$ cd install/face_recognition
$ ./face_recognition data/model/retinaface.rknn data/model/facenet.rknn 33
```

33 is the index of camera device.