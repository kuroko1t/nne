import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 224, 224)
trt_file = 'alexnet_trt.pth'
model = torchvision.models.alexnet(pretrained=True).cuda()

# convert pytorch model to TensorRT model
nne.cv2trt(model, input_shape, trt_file)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# load TensorRT model
trt_model = nne.load_trt(trt_file)

# inference
output_data = nne.infer_trt(trt_model, input_data)

print(output_data)
