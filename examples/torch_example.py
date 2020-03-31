import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 64, 64)
model = torchvision.models.resnet34(pretrained=True).cuda()

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
output_data = nne.infer_torch(model, input_data, benchmark=True)

print(output_data)
