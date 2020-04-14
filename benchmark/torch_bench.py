import torch
import numpy as np
import nne
import os
import torchvision

input_shape = (1, 3, 224, 224)
model = torchvision.models.resnet34(pretrained=True).cuda()

bm = nne.Benchmark(name='torch')
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
output_data = nne.infer_torch(model, input_data, bm=bm)
