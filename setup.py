from setuptools import setup, find_packages

setup(
    name='nne',
    packages=find_packages(),
    install_requires=['torch', 'onnx', 'tensorflow-cpu', 'tensorflow_addons', 'onnx_tf @ git+https://github.com/onnx/onnx-tensorflow', 'torchvision', 'onnxruntime-gpu'],
    version='0.1'
)
