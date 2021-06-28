from nne.quant.onnx import quant_oplist, quant_summary, quantize
import torchvision
import nne

input_shape = (1, 3, 64, 64)
onnx_file = 'resnet.onnx'
model = torchvision.models.resnet34(pretrained=True)

# convert pytorch model to onnx model
nne.cv2onnx(model, input_shape, onnx_file)

# onnx model to quantized model
quantize("resnet.onnx")

# return support quantized operation list
quantie_op = quant_oplist()

# return summary information about quantized model
summary = quant_summary("resnet.quant.onnx")
print(summary)
