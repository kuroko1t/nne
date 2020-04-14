import torch
from .common import *

def cv2torchscript(model, input_shape, script_path):
    model.eval()
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')
    traced = torch.jit.trace(model, dummy_input)
    traced.save(script_path)

def load_torchscript(script_path):
    model = torch.jit.load(script_path)
    return model

def infer_torchscript(model, input_data, bm=None):
    input_data = torch.from_numpy(input_data)
    if check_model_is_cuda(model):
        input_data = input_data.cuda()
    if bm:
        output = bm.measure(model, name='torchscript')(input_data)
    else:
        output = model(input_data)
    return output.detach().cpu().numpy()
