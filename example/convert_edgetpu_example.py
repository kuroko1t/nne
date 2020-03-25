import torch2tflite
import torchvision
import torch
import numpy as np

input_shape = (10, 3, 112, 112)
model = torchvision.models.mobilenet_v2(pretrained=True)#.cuda()
dummy_input = torch.randn(input_shape, device='cpu')

tflite_file = 'mobilenet.tflite'

torch2tflite.convert2tflite(model , dummy_input, tflite_file, edgetpu=True)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

output_data = torch2tflite.infer_tflite(tflite_file, input_data)

print(output_data)
