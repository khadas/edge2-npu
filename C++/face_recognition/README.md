## Compile demo

```sh
$ bash build.sh
```

## Inference demo

This demo integrates retinaface and facenet.

```sh
# Generate feature library
$ cd install/face_recognition
$ ./face_recognition data/model/retinaface.rknn data/model/facenet.rknn 1
```

Feature library will generate in install/face_recognition/data named face_feature_lib.

```sh
# Identification
$ ./face_recognition data/model/retinaface.rknn data/model/facenet.rknn data/model/lin_1.jpg
```

When you generate feature library, please make sure only one face in picture.