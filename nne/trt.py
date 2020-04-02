import torch
from torch2trt import torch2trt, TRTModule
from .common import *
from .benchmark import benchmark as bm

def cv2trt(model, input_shape, trt_file, fp16_mode=False):
    """
    convert torch model to tflite model using onnx
    """
    model.eval()
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')
    model_trt = torch2trt(model, [dummy_input], fp16_mode=fp16_mode, max_batch_size=input_shape[0])
    torch.save(model_trt.state_dict(), trt_file)


def infer_trt(trt_file, input_data, benchmark=False):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_file))
    input_data = torch.from_numpy(input_data).cuda()
    if benchmark:
        output = bm(model_trt)(input_data)
    else:
        output = model_trt(input_data)
    return output.detach().cpu().numpy()
