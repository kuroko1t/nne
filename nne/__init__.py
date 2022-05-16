from .analyze import onnx as onnx_analyze
import json
from .analyze import tflite as tflite_analyze
from .convert.torch import *
from .benchmark import *
from .convert.torchscript import *
from .convert.onnx import *
from .convert.tflite import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if check_tensorrt():
    from .trt import *


def analyze(model_path, output_path=None):
    ext = os.path.splitext(model_path)[1]
    if ext == ".onnx":
        model_info = onnx_analyze.analyze_graph(model_path, output_path)
        return model_info
    elif ext == ".tflite":
        model_info = tflite_analyze.analyze_graph(model_path, output_path)
        if output_path == None:
            output_path = model_path.replace(".tflite", "_tflite.json")
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2, cls=tflite_analyze.NumpyEncoder)
        print(f"Write Dump Result -> {output_path}")
        return model_info
    else:
        raise Exception(f"no support {ext} file")
