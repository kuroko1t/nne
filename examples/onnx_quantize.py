from nne.quant.onnx import quant_oplist, quant_summary, quantize

# onnx model to quantized model
quantize("resnet.onnx")

# return support quantized operation list
quantie_op = quant_oplist()

# return summary information about quantized model
summary = quant_summary("resnet.quant.onnx")
print(summary)
