# torch2tflite

convert pytorch model to tflite model  
Please feel free to issue and PR.

## Install

```bash
python -m pip install -e . 
```

## example

create .tflite from pytorch model

```python3
import torch2tflite
import torchvision
import torch

model = torchvision.models.alexnet(pretrained=True).cuda()
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')

torch2tflite.convert2tflite(model , dummy_input, 'alexnet.tflite')
```

create .tflite(quantize) for edge tpu

```python3
input_shape = (10, 3, 112, 112)
model = torchvision.models.mobilenet_v2(pretrained=True)
dummy_input = torch.randn(input_shape, device='cpu')

tflite_file = 'mobilenet.tflite'

torch2tflite.convert2tflite(model , dummy_input, tflite_file, edgetpu=True)
```


## Support Format

|format  | support  |
|---|---|
| tflite  |  :white_check_mark: |
| edge tpu  | trial  |
| onnx||
| tensorRT||

