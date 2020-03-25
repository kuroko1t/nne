import torch2tflite
import torchvision
import torch
import numpy as np

input_shape = (10, 3, 224, 224)
model = torchvision.models.alexnet(pretrained=True).cuda()
dummy_input = torch.randn(input_shape, device='cuda')

tflite_file = 'alexnet.tflite'

torch2tflite.convert2tflite(model , dummy_input, tflite_file, edgetpu=True)

input_data = np.array(np.random.random_sample(input_shape), type=np.float32)

output_data = torch2tflite.infer_tflite(tflite_file, input_dta)

print(output_data)
