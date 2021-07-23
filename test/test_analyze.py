import unittest
import nne
import torchvision
import torch
import numpy as np
from nne.quant.onnx import quant_oplist, quant_summary, quantize

class AnalyzeTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AnalyzeTests, self).__init__(*args, **kwargs)

    def test_onnx(self):
        input_shape = (1, 3, 64, 64)
        onnx_file = 'resnet.onnx'
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2onnx(model, input_shape, onnx_file)
        nne.analyze(onnx_file)
        nne.analyze(onnx_file, "resnet.json")
