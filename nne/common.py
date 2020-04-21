import platform
import os
import onnxsim

def check_jetson():
    if platform.machine() == "aarch64":
        return True
    else:
        return False

def check_tensorrt():
    try:
        import tensorrt
        return True
    except:
        return False

def check_model_is_cuda(model):
    return next(model.parameters()).is_cuda

def onnx_simplify(model, input_shapes):
    model_opt, check_ok = onnxsim.simplify(
        model, check_n=3, perform_optimization=False, skip_fuse_bn=False, input_shapes=None)
    return model_opt, check_ok
