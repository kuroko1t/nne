from setuptools import setup, find_packages
import torch

def check_gpu_enable():
    return torch.cuda.is_available()

def get_requires():
    requires = ['onnx', 'tensorflow-cpu', 'tensorflow_addons', 'onnx_tf @ git+https://github.com/onnx/onnx-tensorflow', 'torchvision']
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
