import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 224, 224)
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()
dummy_input = torch.randn(input_shape, device='cuda')

tflite_file = 'mobilenet.tflite'

nne.cv2tflite(model , dummy_input, tflite_file)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

output_data = nne.infer_tflite(tflite_file, input_data)

print(output_data)


