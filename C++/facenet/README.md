## Compile demo

```sh
$ bash build.sh
```

## Inference demo

```sh
# Generate feature library
$ cd install/facenet
$ ./facenet data/model/facenet.rknn 1

# Identification
$ ./facenet data/model/facenet.rknn data/model/lin_1.jpg
```