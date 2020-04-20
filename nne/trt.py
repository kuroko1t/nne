import torch
import tensorrt as trt
import pycuda.driver as cuda
from .common import *
from .onnx import *

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
        with open(onnx_file, 'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        with open(trt_file, "wb") as f:
            f.write(engine.serialize())
    #if check_model_is_cuda(model):
    #    dummy_input = torch.randn(input_shape, device='cuda')
    #else:
    #    dummy_input = torch.randn(input_shape, device='cpu')
    #model_trt = torch2trt(model, [dummy_input], fp16_mode=fp16_mode, max_batch_size=input_shape[0])
    #torch.save(model_trt.state_dict(), trt_file)


def load_trt(trt_file):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def infer_trt(model, input_data, benchmark=False, bm=None):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
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
    #input_data = torch.from_numpy(input_data).cuda()
    #if bm:
    #    output = bm.measure(model, name='TensorRT')(input_data)
    #else:
    #    output = model(input_data)
    #return output.detach().cpu().numpy()
