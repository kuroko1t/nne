import platform
import os

def check_jetson():
    if platform.machine() == 'aarch64':
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
