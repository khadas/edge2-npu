# Python NPU demo for Edge2

### Run

use resnet18 as example:

```sh
$ cd resnet18
$ python3 resnet18.py 
--> Load RKNN model
done
--> Init runtime environment
I RKNN: [17:07:10.282] RKNN Runtime Information: librknnrt version: 1.3.0 (c193be371@2022-05-04T20:16:33)
I RKNN: [17:07:10.282] RKNN Driver Information: version: 0.7.2
I RKNN: [17:07:10.282] RKNN Model Information: version: 1, toolkit version: 1.3.0-11912b58(compiler version: 1.3.0 (c193be371@2022-05-04T20:23:58)), target: RKNPU v2, target platform: rk3588, framework name: PyTorch, framework layout: NCHW
done
--> Running model
resnet18
-----TOP 5-----
[812]: 0.9996383190155029
[404]: 0.00028062646742910147
[657]: 1.632110434002243e-05
[833 895]: 1.015904672385659e-05
[833 895]: 1.015904672385659e-05

done
```
