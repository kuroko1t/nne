# nne

<p align="center"><img width="40%" src="docs/logo.png" /></p>

convert pytorch model for Edge Device

contents

- [Install](#install)
- [Example](#Example)
  - [onnx](#onnx)
  - [tflite](#tflite)
  - [tflite(edgetpu)](#tflite-edgetpu)

## Install

```bash
python -m pip install -e . 
```

If you want to compile pytorch model for edgetpu, install edgetpu_compiler [ref](https://coral.ai/docs/edgetpu/compiler/)

## Example

### onnx

```python3
import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 64, 64)
onnx_file = 'resnet.onnx'
model = torchvision.models.resnet34(pretrained=True).cuda()
dummy_input = torch.randn(input_shape, device='cuda')

nne.cv2onnx(model , dummy_input, onnx_file)
```

### tflite

```python3
import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 224, 224)
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
dummy_input = torch.randn(input_shape, device='cuda')

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model , dummy_input, tflite_file)
```

### tflite(edgetpu)

```python3
import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 112, 112)
model = torchvision.models.mobilenet_v2(pretrained=True)
dummy_input = torch.randn(input_shape, device='cpu')

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model , dummy_input, tflite_file, edgetpu=True)
```

## Support Format

|format  | support  |
|---|---|
| tflite  | :white_check_mark: |
| edge tpu  | trial  |
| onnx| :white_check_mark: |
| tensorRT||

## License
MIT
