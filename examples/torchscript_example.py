import nne
import torchvision
import torch
import numpy as np

input_shape = (1, 3, 224, 224)
model = torchvision.models.resnet50(pretrained=True).cuda()
script_file = 'resnet_script.zip'
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
nne.cv2torchscript(model, input_shape, script_path)
nne.infer_torchscript(script_path, input_data, benchmark=True)
