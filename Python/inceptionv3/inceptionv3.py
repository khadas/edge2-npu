import numpy as np
import cv2
import os
import time
import sys
from rknnlite.api import RKNNLite

IMG_PATH = './data/img/goldfish_299x299.jpg'
INPUT_SIZE = 299


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'inception_v3\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)

if __name__ == '__main__':
    # Create RKNN object
    rknn_lite = RKNNLite()

    # load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn('./data/model/inception_v3_quant_frozen.rknn')
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn_lite.inference(inputs=[img])
    x = outputs[0]
    output = np.exp(x)/np.sum(np.exp(x))
    outputs = [output]
    show_outputs(outputs)
    print('done')

    rknn_lite.release()
