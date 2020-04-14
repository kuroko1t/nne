import unittest
import nne
import torchvision
import torch
import numpy as np

class TorchTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TorchTests, self).__init__(*args, **kwargs)

    def test_torch(self):
        input_shape = (1, 3, 64, 64)
        script_file = 'resnet_script.zip'
        model = torchvision.models.resnet34(pretrained=True)
        nne.cv2torchscript(model, input_shape, script_file)

        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        model_script = nne.load_torchscript(script_file)
        out_script = nne.infer_torchscript(model_script, input_data)
        model.eval()
        out_pytorch = model(torch.from_numpy(input_data)).detach().cpu().numpy()
        np.testing.assert_allclose(out_script, out_pytorch, rtol=1e-03, atol=1e-05)

if __name__ == "__main__":
    unittest.main()
