from .convert.tflite import *
from .convert.onnx import *
from .convert.torchscript import *
from .benchmark import *
from .convert.torch import *
#from .analize.onnx import analize_graph
from .analize import onnx as onnx_analize
if check_tensorrt():
    from .trt import *


def analyze(model_path, output_path):
    ext = os.path.splitext(model_path)[1]
    if ext == ".onnx":
        onnx_analize.analyze_graph(model_path, output_path)
    else:
        raise Exception(f"no support {ext} file")
