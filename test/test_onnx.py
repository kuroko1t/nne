import unittest
import nne
import torchvision
import torch
import numpy as np
from nne.quant.onnx import quant_oplist, quant_summary, quantize

class OnnxTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OnnxTests, self).__init__(*args, **kwargs)

    def test_onnx(self):
        input_shape = (1, 3, 64, 64)
        onnx_file = 'resnet.onnx'
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2onnx(model, input_shape, onnx_file)

        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        model_onnx = nne.load_onnx(onnx_file)
        out_onnx = nne.infer_onnx(model_onnx, input_data)
        model.eval()
        out_pytorch = model(torch.from_numpy(input_data)).detach().cpu().numpy()
        np.testing.assert_allclose(out_onnx, out_pytorch, rtol=1e-03, atol=1e-05)

    def test_onnx_quant(self):
        input_shape = (1, 3, 64, 64)
        onnx_file = 'resnet.onnx'
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2onnx(model, input_shape, onnx_file)
        quantize("resnet.onnx")
        quantie_op = quant_oplist()
        summary = quant_summary("resnet.quant.onnx")

if __name__ == "__main__":
    unittest.main()
