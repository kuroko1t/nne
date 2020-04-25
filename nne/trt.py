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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from .common import *
from .onnx import *
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def cv2trt(model, input_shape, trt_file, fp16_mode=False):
    """
    convert torch model to tflite model using onnx
    """
    model.eval()
    onnx_file = os.path.splitext(trt_file)[0] + ".onnx"
    cv2onnx(model, input_shape, onnx_file)
    EXPLICIT_BATCH = 1
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file, "rb") as onnx_model:
        parser.parse(onnx_model.read())
    if parser.num_errors > 0:
        error = self.parser.get_error(0)
        raise Exception(error)
    max_workspace_size = 1 << 28
    max_batch_size = 32
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode
    engine = builder.build_cuda_engine(network)
    with open(trt_file, "wb") as f:
        f.write(engine.serialize())
    os.remove(onnx_file)


def load_trt(trt_file):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_file, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def infer_trt(engine, input_data, bm=None):
    #outputs = engine.run(input_data)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)

    def execute():
        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        with engine.create_execution_context() as context:
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, h_input, stream)
            # Run inference.
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()
            # Return the host output.
        return h_output

    if bm:
        output = bm.measure(execute, name="TensorRT")()
    else:
        output = execute()
    return output
