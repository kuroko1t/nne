from setuptools import setup, find_packages
import platform
import torch
import os

def check_tensorrt():
    os.environ['LD_LIBRARY_PATH'] = "{os.environ['LD_LIBRARY_PATH']}:/usr/local/lib/64"
    try:
        import tensorrt
        return True
    except:
        return False

def check_gpu_enable():
    return torch.cuda.is_available()

def get_requires():
    requires = ['onnx', 'tensorflow-cpu', 'tensorflow_addons', 'onnx_tf @ git+https://github.com/onnx/onnx-tensorflow', 'torchvision', 'matplotlib']
    if check_tensorrt():
        requires += ['torch2trt @ git+https://github.com/NVIDIA-AI-IOT/torch2trt']
    else:
        if check_gpu_enable():
            requires += ['onnxruntime-gpu']
        else:
            requires += ['onnxruntime']
    return requires

setup(
    name='nne',
    packages=find_packages(),
    install_requires=get_requires(),
    version='0.1'
)
