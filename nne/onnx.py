#MIT License
#
#Copyright (c) 2020 kurosawa
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import onnx
import torch
from .common import *
import sys
if not check_jetson():
    import onnxruntime

def cv2onnx(model, input_shape, onnx_file):
    """
    convert torch model to tflite model using onnx
    """
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device="cuda")
    else:
        dummy_input = torch.randn(input_shape, device="cpu")
    try:
        torch.onnx.export(model, dummy_input, onnx_file,
                          do_constant_folding=True,
                          input_names=[ "input" ] , output_names=["output"])
        onnx_model = onnx.load(onnx_file)
    except RuntimeError as e:
        opset_version=11
        if "aten::upsample_bilinear2d" in e.args[0]:
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            torch.onnx.export(model, dummy_input, onnx_file, verbose=True,
                              input_names=[ "input" ] , output_names=["output"],
                              opset_version = opset_version,
                              operator_export_type=operator_export_type)
            onnx_model = onnx.load(onnx_file)
            onnx.checker.check_model(onnx_model)
        else:
            print("[ERR]:",e)
            sys.exit()
    except Exception as e:
        print("[ERR]:", e)
        sys.exit()
    model_opt, check_ok = onnx_simplify(onnx_model, input_shape)
    if check_ok:
        onnx.save(model_opt, onnx_file)


def load_onnx(onnx_file):
    ort_session = onnxruntime.InferenceSession(onnx_file)
    return ort_session


def infer_onnx(ort_session, input_data, bm=None):
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    if bm:
        ort_outs = bm.measure(ort_session.run, name="onnx")(None, ort_inputs)
    else:
        ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]
