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
from onnx_tf.backend import prepare
import torch
import tensorflow as tf
import os
import shutil
import sys
import subprocess
from .common import *
from .onnx import cv2onnx
import numpy as np

def onnx2tflite(mdoel, tflite_path):
    onnx_model = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model)
    tmp_pb_file = "tmp.pb"
    tf_rep.export_graph(tmp_pb_file)
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_pb_file)
    tflite_model = converter.convert()
    shutil.rmtree(tmp_pb_file)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

def cv2tflite(model, input_shape, tflite_path, edgetpu=False, quantization=False):
    """
    convert torch model to tflite model using onnx
    """
    onnx_input_flag = False
    if type(model) == str:
        ext = os.path.splitext(model)[1]
        if ext == ".onnx":
            onnx_input_flag = True
    tmp_pb_file = "tmp.pb"
    if not onnx_input_flag:
        onnx_file = "tmp.onnx"
        cv2onnx(model, input_shape, onnx_file)
    else:
        onnx_file = model
    onnx_model = onnx.load(onnx_file)
    onnx_input_names = [input.name for input in onnx_model.graph.input]
    onnx_output_names = [output.name for output in onnx_model.graph.output]
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tmp_pb_file)

    converter = tf.lite.TFLiteConverter.from_saved_model(tmp_pb_file)

    if quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if edgetpu:
        if type(input_shape[0]) == tuple:
            if check_model_is_cuda(model):
                dummy_input = tuple([np.randn(ishape) for ishape in input_shape])
            else:
                dummy_input = tuple([np.randn(ishape) for ishape in input_shape])
        elif type(input_shape) == tuple:
            if check_model_is_cuda(model):
                dummy_input = np.randn(input_shape)
            else:
                dummy_input = np.randn(input_shape)
        else:
            raise Exception("input_shape must be tuple")
        train = tf.convert_to_tensor(input_data)
        my_ds = tf.data.Dataset.from_tensor_slices((train)).batch(10)
        def representative_dataset_gen():
            for input_value in my_ds.take(10):
                yield [input_value]
        converter.representative_dataset = representative_dataset_gen
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # convert tensorflow to tflite model
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    if not onnx_input_flag:
        os.remove(onnx_file)
    shutil.rmtree(tmp_pb_file)

    if edgetpu:
        subprocess.check_call(f"edgetpu_compiler {tflite_path}", shell=True)


def load_tflite(tflitepath):
    interpreter = tf.lite.Interpreter(model_path=tflitepath)
    # allocate memory
    interpreter.allocate_tensors()
    return interpreter


def infer_tflite(interpreter, input_data, bm=None):
    # get model input and output propaty
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ## get input shape
    #input_shape = input_details[0]["shape"]
    def execute():
        # set tensor pointer to index
        if type(input_data) == tuple:
            for i, data in enumerate(input_data):
                interpreter.set_tensor(input_details[i]["index"], data)
        else:
            interpreter.set_tensor(input_details[0]["index"], input_data)
        # execute infer
        interpreter.invoke()
        # get result from index of output_details
        output_data = interpreter.get_tensor(output_details[0]["index"])
        return output_data
    if bm:
        output_data = bm.measure(execute, name="tflite")()
    else:
        output_data = execute()
    return output_data
