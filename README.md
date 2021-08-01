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
- [Script](#Script)
  - [Model Summary](#Model-Summary)
- [Support Format](#Support-Format)
- [License](#License)

## Install

```bash
python -m pip install -e .
```

* edgetpu

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

## Script

* convert onnx model to tflite

```bash
$nne -h

usage: nne [-h] [-a ANALYZE] [-s SIMPLYFY_PATH] [-t TFLITE_PATH] model_path

Neural Network Graph Analyzer

positional arguments:
  model_path            model path for analyzing

optional arguments:
  -h, --help            show this help message and exit
  -a ANALYZE, --analyze ANALYZE
                        Specify the path to output the Node information of the model in json format.
  -s SIMPLYFY_PATH, --simplyfy_path SIMPLYFY_PATH
                        onnx model to simplyfier
  -t TFLITE_PATH, --tflite_path TFLITE_PATH
                        onnx model to tflite
```

### Model Summary

this script only support onnx model yet.

* output summarized model information.

```bash
$ nne-analizer conv.onnx

#### SUMMARY ONNX MODEL ####
opset: 9
INPUT: [[1, 3, 64, 64]]
OUTPUT: [[1, 1000]]
--Node List-- num(89)
{'Conv': 36, 'Relu': 33, 'MaxPool': 1, 'Add': 16, 'GlobalAveragePool': 1, 'Flatten': 1, 'Gemm': 1}
```

* dump detailed model information(node name, attrs) to json file.

```bash
$ nne-analizer resnet.onnx -o resnet.json
```


## Support Format

|format  | support  |
|---|---|
| tflite  | :white_check_mark: |
| edge tpu  | trial  |
| onnx| :white_check_mark: |
| tensorRT| :white_check_mark: |

## License
Apache 2.0
