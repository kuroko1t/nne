import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 64, 64)
onnx_file = 'resnet.onnx'
model = torchvision.models.resnet34(pretrained=True).cuda()

nne.cv2onnx(model, input_shape, onnx_file)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

onnx_model = nne.load_onnx(onnx_file)

output_data = nne.infer_onnx(onnx_model, input_data)

print(output_data)
