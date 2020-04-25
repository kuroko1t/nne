from setuptools import setup, find_packages
import platform
import torch
import os
import shutil
import subprocess

TENSORRT_ROOT = "/nnet/TensorRT/"
CUDA_INCLUDE_PATH = "/usr/local/cuda/include/"

def check_tensorrt():
    try:
        import tensorrt
        return True
    except:
        return False

def check_gpu_enable():
    return torch.cuda.is_available()

def remove_onnxtensorrt(current_dir):
    os.chdir(current_dir)
    shutil.rmtree("onnx-tensorrt")
    return

def install_onnxtensorrt():
    if not shutil.which("swig"):
        raise Exception("Please Install swig")
    if not shutil.which("protoc"):
        raise Exception("Please Install protobuf")
    res = subprocess.call("git clone --recursive https://github.com/onnx/onnx-tensorrt", shell=True)
    if res:
        remove_onnxtensorrt(current_dir)
        raise Exception("git clone Exception")
    current_dir = os.getcwd()
    os.chdir("onnx-tensorrt")
    patch_tensorrt_file()
    try:
        build_onnx_tensorrt()
    except Exception as e:
        remove_onnxtensorrt(current_dir)
        raise Exception(e)

    res = subprocess.call("python3 setup.py install", shell=True)
    if res:
        remove_onnxtensorrt(current_dir)
        raise Exception("onnx-tensorrt install Exception")
    remove_onnxtensorrt(current_dir)

def patch_tensorrt_file():
    set_define = """#ifdef TENSORRT_BUILD_LIB
    #ifdef _MSC_VER
    #define TENSORRTAPI __declspec(dllexport)
    #else
    #define TENSORRTAPI __attribute__((visibility("default")))
    #endif
    #else
    #define TENSORRTAPI
    #endif"""
    with open("NvOnnxParser.h",'r') as f:
        orig_file = f.read()

    with open("NvOnnxParser.h", "w") as f:
        f.write(set_define+orig_file)

def build_onnx_tensorrt():
    current_dir = os.getcwd()
    subprocess.call("mkdir build", shell=True)
    os.chdir("build")
    res = subprocess.call(f"cmake -DCUDA_INCLUDE_DIRS={CUDA_INCLUDE_PATH} ..", shell=True)
    if res:
        raise Exception("onnx-tensorrt cmake failed")
    res = subprocess.call(f"make -j$(nproc)", shell=True)
    if res:
        raise Exception("onnx-tensorrt build failed")
    os.chdir(current_dir)

def get_requires():
    requires = ["onnx", "tensorflow-cpu", "tensorflow_addons", "onnx_tf @ git+https://github.com/onnx/onnx-tensorflow", "matplotlib", "onnx-simplifier"]
    if check_tensorrt():
        try:
            import onnx_tensorrt
        except:
            install_onnxtensorrt()
        requires += ["pycuda"]
    else:
        if check_gpu_enable():
            requires += ["onnxruntime-gpu"]
        else:
            requires += ["onnxruntime"]
    return requires

setup(
    name="nne",
    packages=find_packages(),
    install_requires=get_requires(),
    version="0.1"
 )
