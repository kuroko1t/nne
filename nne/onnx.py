# Copyright 2020 kurosawa. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import onnx
import torch
from .common import *
import sys
try:
    import onnxruntime
except:
    pass


def cv2onnx(model, input_shape, onnx_file, simplify=False):
    """
    convert torch model to tflite model using onnx
    """
    if type(input_shape[0]) == tuple:
        if check_model_is_cuda(model):
            dummy_input = tuple([torch.randn(ishape, device="cuda") for ishape in input_shape])
        else:
            dummy_input = tuple([torch.randn(ishape, device="cpu") for ishape in input_shape])
    elif type(input_shape) == tuple:
        if check_model_is_cuda(model):
            dummy_input = torch.randn(input_shape, device="cuda")
        else:
            dummy_input = torch.randn(input_shape, device="cpu")
    else:
        raise Exception("input_shape must be tuple")

    try:
        torch.onnx.export(model, dummy_input, onnx_file,
                          input_names=[ "input" ] , output_names=["output"])
    except RuntimeError as e:
        opset_version = 12
        torch.onnx.export(model, dummy_input, onnx_file,
                          opset_version=opset_version,
                          input_names=[ "input" ] , output_names=["output"])

    if simplify:
        onnx_model = load_onnx(onnx_file)
        model_opt, check_ok = onnx_simplify(onnx_model, input_shape)
        if check_ok:
            onnx.save(model_opt, onnx_file)

def load_onnx(onnx_file):
    sess = onnxruntime.InferenceSession(onnx_file)
    if "TensorrtExecutionProvider" in sess.get_providers():
        print("onnxmodel with TensorrtExecutionProvider")
        sess.set_providers(["TensorrtExecutionProvider"])
    elif "CUDAExecutionProvider" in sess.get_providers():
        print("onnxmodel with CUDAExecutionProvider")
        sess.set_providers(["CUDAExecutionProvider"])
    elif "CPUExecutionProvider" in sess.get_providers():
        print("onnxmodel with CPUExecutionProvider")
        sess.set_providers(["CPUExecutionProvider"])
    return sess


def infer_onnx(sess, input_data, bm=None):
    if type(input_data) == tuple:
        for i, data in enumerate(input_data):
            if i == 0:
                ort_inputs = {sess.get_inputs()[i].name: data}
            else:
                ort_inputs[sess.get_inputs()[i].name] = data
    else:
        ort_inputs = {sess.get_inputs()[0].name: input_data}
    if bm:
        ort_outs = bm.measure(sess.run, name="onnx")(None, ort_inputs)
    else:
        ort_outs = sess.run(None, ort_inputs)
    return ort_outs[0]
