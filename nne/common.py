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

import platform
import os
import onnxsim

def check_jetson():
    if platform.machine() == "aarch64":
        return True
    else:
        return False

def check_tensorrt():
    try:
        import tensorrt
        return True
    except:
        return False

def check_model_is_cuda(model):
    return next(model.parameters()).is_cuda

def onnx_simplify(model, input_shapes):
    model_opt, check_ok = onnxsim.simplify(
        model, check_n=3, perform_optimization=False, skip_fuse_bn=False, input_shapes=None)
    return model_opt, check_ok
