from setuptools import setup, find_packages
import platform
import torch
import os

def check_tensorrt():
    try:
        import tensorrt
        return True
    except:
        return False

def check_jetson():
    if platform.machine() == "aarch64":
        return True
    else:
        return False

def check_gpu_enable():
    return torch.cuda.is_available()

def get_requires():
    if check_jetson():
        requires = ["tensorflow"]
    else:
        requires = ["tensorflow-cpu", "tensorflow_addons"]
    requires += ["onnx", "onnx_tf @ git+https://github.com/onnx/onnx-tensorflow", "matplotlib", "onnx-simplifier"]
    if check_tensorrt():
        requires += ["pycuda"]
    return requires

setup(
    name="nne",
    packages=find_packages(),
    install_requires=get_requires(),
    version="0.1"
)
