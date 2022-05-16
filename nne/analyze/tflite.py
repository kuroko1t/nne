import tensorflow as tf
import json
import numpy as np


def analyze_graph(model_path, output_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    graphinfo = interpreter.get_tensor_details()
    for i, op in enumerate(graphinfo):
        graphinfo[i]["shape"] = op["shape"].tolist()
        graphinfo[i]["shape_signature"] = op["shape_signature"].tolist()
        graphinfo[i]["dtype"] = [op["dtype"].__name__]
        graphinfo[i]["quantization_parameters"]["scales"] = op["quantization_parameters"]["scales"].tolist()
        graphinfo[i]["quantization_parameters"]["zero_points"] = op["quantization_parameters"]["zero_points"].tolist()
    return graphinfo


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
