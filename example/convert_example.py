import torch2tflite
import torchvision
import torch

model = torchvision.models.alexnet(pretrained=True).cuda()
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')

torch2tflite.convert2tflite(model , dummy_input, 'alexnet.tflite')


