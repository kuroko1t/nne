from .convert.tflite import *
from .convert.onnx import *
from .convert.torchscript import *
from .benchmark import *
from .convert.torch import *
from .analyze import onnx as onnx_analyze
if check_tensorrt():
    from .trt import *


def analyze(model_path, output_path=None):
    ext = os.path.splitext(model_path)[1]
    if ext == ".onnx":
        model_info = onnx_analyze.analyze_graph(model_path, output_path)
        return model_info
    else:
        raise Exception(f"no support {ext} file")
