from setuptools import setup, find_packages
import platform
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

def get_requires():
    if check_jetson():
        requires = ["tensorflow"]
    else:
        requires = ["tensorflow", "tensorflow_addons"]
    requires += ["torch", "onnx", "onnx_tf @ git+https://github.com/onnx/onnx-tensorflow",
                 "matplotlib", "onnx-simplifier"]
    if check_tensorrt():
        requires += ["pycuda"]
    return requires

setup(
    name="nne",
    scripts=["bin/nne"],
    packages=find_packages(),
    install_requires=get_requires(),
    version="0.1"
)
