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

import torch
from .common import *

def cv2torchscript(model, input_shape, script_path):
    model.eval()
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device="cuda")
    else:
        dummy_input = torch.randn(input_shape, device="cpu")
    traced = torch.jit.trace(model, dummy_input)
    traced.save(script_path)

def load_torchscript(script_path):
    model = torch.jit.load(script_path)
    return model

def infer_torchscript(model, input_data, bm=None):
    input_data = torch.from_numpy(input_data)
    if check_model_is_cuda(model):
        input_data = input_data.cuda()
    if bm:
        output = bm.measure(model, name="torchscript")(input_data)
    else:
        output = model(input_data)
    return output.detach().cpu().numpy()
