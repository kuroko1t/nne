import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 112, 112)
model = torchvision.models.mobilenet_v2(pretrained=True)
dummy_input = torch.randn(input_shape, device='cpu')

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model , dummy_input, tflite_file, edgetpu=True)

