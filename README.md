# torch2tflite

convert pytorch model to tflite model

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