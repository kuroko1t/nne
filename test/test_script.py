import unittest
import nne
import torchvision
import torch
import numpy as np
import subprocess

class ScriptTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ScriptTests, self).__init__(*args, **kwargs)
        self.onnx_file = 'resnet.onnx'
        input_shape = (1, 3, 64, 64)
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2onnx(model, input_shape, self.onnx_file)

    def test_analyze(self):
        subprocess.check_call(["nne", self.onnx_file])
        subprocess.check_call(["nne", self.onnx_file, "-a", "resnet.json"])

    def test_convert(self):
        subprocess.check_call(["nne", self.onnx_file, "-s", "resnet_smip.onnx"])
        subprocess.check_call(["nne", self.onnx_file, "-t", "resnet.tflite"])
