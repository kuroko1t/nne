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
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from .common import *
from .onnx import *
import numpy as np

def cv2trt(model, input_shape, trt_file, fp16_mode=False):
    """
    convert torch model to tflite model using onnx
    """
    model.eval()
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    onnx_file = os.path.splitext(trt_file)[0] + ".onnx"
    cv2onnx(model, input_shape, onnx_file)
    EXPLICIT_BATCH = 1
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file, "rb") as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        with open(trt_file, "wb") as f:
            f.write(engine.serialize())
    os.remove(onnx_file)


def load_trt(trt_file):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def infer_trt(engine, input_data, bm=None):
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
