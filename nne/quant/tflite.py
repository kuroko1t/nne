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

import tensorflow as tf
import collections


def quantize(modelpath, input_shape):
    """convert model to quantized tflite model"""
    onnx_file = "tmp.onnx"
    ext = os.path.splitext(os.path.basename(modelpath))[1]
    if ext == "onnx":
        cv2onnx(modelpath, input_shape, onnx_file)
    model_quant = modelpath.replace(".onnx", ".quant.onnx")
    quantized_model = quantize_qat(modelpath, model_quant)

def quant_oplist():
    qoplist = {}
    qoplist.update(QLinearOpsRegistry)
    qoplist.update(QDQRegistry)
    qoplist.update(IntegerOpsRegistry)
    quantized_op = {}
    for v in qoplist.values():
        quantized_op.update({v.__name__:v})
    #quantized_opname.append("DynamicQuantizeLinear")
    #print(qoplist)
    return quantized_op

def quant_summary(quantmodel):
    summary = {}
    model = onnx.load(quantmodel)
    quant_op = []
    summary.update({"opset_version":model.opset_import[-1].version})
    for node in model.graph.node:
        if node.op_type in quant_oplist().keys():
            quant_op.append(node.op_type)
    quant_op_counter = dict(collections.Counter(quant_op))
    summary.update({"quant_op":quant_op_counter})
    return summary
