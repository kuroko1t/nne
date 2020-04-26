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

def infer_torch(model, input_data,  bm=None):
    """
    model : loaded model
    input_data: numpy array
    """
    model.eval()
    input_data = torch.from_numpy(input_data)
    if check_model_is_cuda:
        input_data = input_data.cuda()
    if bm:
        output = bm.measure(model, name="torch")(input_data)
    else:
        output = model(input_data)
    return output.detach().cpu().numpy()
