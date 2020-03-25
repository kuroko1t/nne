from setuptools import setup, find_packages

setup(
    name='torch2tflite',
    packages=find_packages(),
    install_requires=['torch', 'onnx', 'tensorflow-cpu', 'onnx_tf @ https://github.com/onnx/onnx-tensorflow', 'torchvision'],
    version='0.1'
)
