from setuptools import setup, find_packages

setup(
    name='torch2tflite',
    packages=find_packages(),
    install_requires=['torch', 'onnx', 'tensorflow', 'onnx_tf', 'torchvision'],
    version='0.1'
)
