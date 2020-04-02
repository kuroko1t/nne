import torch
from .benchmark import benchmark as bm
from .common import *

def infer_torch(model, input_data,  benchmark=False):
    """
    model : loaded model
    input_data: numpy array
    """
    model.eval()
    input_data = torch.from_numpy(input_data)
    if check_model_is_cuda:
        input_data = input_data.cuda()
    if benchmark:
        output = bm(model)(input_data)
    else:
        output = model(input_data)
    return output.detach().cpu().numpy()
