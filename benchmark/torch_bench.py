import torch
import numpy as np
import nne
import os
import torchvision

input_shape = (1, 3, 224, 224)
model = torchvision.models.resnet34(pretrained=True)
torch.save(model, "resnet.pt")

bm = nne.Benchmark(name='torch')
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
output_data = nne.infer_torch(model, input_data, bm=bm)

## onnx
onnx_file = "resnet.onnx"
bm = nne.Benchmark(name='onnx')
nne.cv2onnx(model, input_shape, onnx_file)
onnx_model = nne.load_onnx(onnx_file)
nne.infer_onnx(onnx_model, input_data, bm=bm)

## tflite
tflite_file = "resnet.tflite"
bm = nne.Benchmark(name='tflite')
nne.cv2tflite(model, input_shape, tflite_file)
tflite_model = nne.load_tflite(tflite_file)
nne.infer_tflite(tflite_model, input_data, bm=bm)
