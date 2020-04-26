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
