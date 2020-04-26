import torchvision
import torch
import numpy as np
import nne

input_shape = (10, 3, 224, 224)
model = torchvision.models.mobilenet_v2(pretrained=True).cuda()

tflite_file = 'mobilenet.tflite'

# convert pytorch model to tensorflow lite
nne.cv2tflite(model, input_shape, tflite_file)

input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# load tflite model
tflite_model = nne.load_tflite(tflite_file)

# inference
output_data = nne.infer_tflite(tflite_model, input_data)

print(output_data)
