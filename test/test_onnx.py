import unittest
import nne
import torchvision
import torch
import numpy as np

class OnnxTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(OnnxTests, self).__init__(*args, **kwargs)

    def test_onnx(self):
        input_shape = (1, 3, 64, 64)
        onnx_file = 'resnet.onnx'
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2onnx(model, input_shape, onnx_file)

        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        out_onnx = nne.infer_onnx(onnx_file, input_data)
        model.eval()
        out_pytorch = model(torch.from_numpy(input_data)).detach().cpu().numpy()
        np.testing.assert_allclose(out_onnx, out_pytorch, rtol=1e-03, atol=1e-05)

if __name__ == "__main__":
    unittest.main()
