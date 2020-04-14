<p align="center"><img width="40%" src="docs/logo.png" /></p>

convert pytorch model for Edge Device

# nne
contents

- [Install](#install)
- [Example](#Example)
  - [onnx](#onnx)
  - [tflite](#tflite)
  - [tflite(edgetpu)](#tflite-edgetpu)
  - [TensorRT](#tensorrt)
- [Support Format](#Support-Format)
- [License](#License)

## Install

```bash
python -m pip install -e .
```

If you want to compile pytorch model for edgetpu, [install edgetpu_compiler](https://coral.ai/docs/edgetpu/compiler/)

## Example

example compile pytorch model for edge device. See [example](https://github.com/kuroko1t/nne/tree/master/examples) for details

### onnx

comvert to onnx model

```python3
import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 64, 64)
onnx_file = 'resnet.onnx'
model = torchvision.models.resnet34(pretrained=True).cuda()

nne.cv2onnx(model, input_shape, onnx_file)
```

### tflite

comvert to tflite model

```python3
import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 224, 224)
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model, input_shape, tflite_file)
```

### tflite(edgetpu)

comvert to tflite model(edge tpu)

```python3
import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 112, 112)
model = torchvision.models.mobilenet_v2(pretrained=True)

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model, input_shape, tflite_file, edgetpu=True)
```

### TensorRT

convert to TensorRT model

```python3
import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 224, 224)
trt_file = 'alexnet_trt.pth'
model = torchvision.models.alexnet(pretrained=True).cuda()
nne.cv2trt(model, input_shape, trt_file)
```

## Support Format

|format  | support  |
|---|---|
| tflite  | :white_check_mark: |
| edge tpu  | trial  |
| onnx| :white_check_mark: |
| tensorRT| :white_check_mark: |

## License
MIT
