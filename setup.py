from setuptools import setup, find_packages

setup(
    name='torch2tflite',
    packages=find_packages(),
    install_requires=['torch', 'onnx', 'tensorflow==1.15.2', 'onnx_tf', 'torchvision'],
    version='0.1'
)
