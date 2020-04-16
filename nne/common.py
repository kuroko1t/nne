import platform
import os

def check_jetson():
    if platform.machine() == 'aarch64':
        return True
    else:
        return False

def check_tensorrt():
    os.environ['LD_LIBRARY_PATH'] = "{os.environ['LD_LIBRARY_PATH']}:/usr/local/lib/64"
    try:
        import tensorrt
        return True
    except:
        return False

def check_model_is_cuda(model):
    return next(model.parameters()).is_cuda
