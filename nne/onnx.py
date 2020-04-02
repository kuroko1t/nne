import onnx
import torch
from .common import *
from .benchmark import benchmark as bm
if not check_jetson():
    import onnxruntime

def cv2onnx(model, input_shape, onnx_file):
    """
    convert torch model to tflite model using onnx
    """
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')
    try:
        torch.onnx.export(model, dummy_input, onnx_file,
                          do_constant_folding=True,
                          input_names=[ "input" ] , output_names=['output'])
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)
    except RuntimeError as e:
        opset_version=11
        if 'aten::upsample_bilinear2d' in e.args[0]:
            operator_export_type = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
            torch.onnx.export(model, dummy_input, onnx_file, verbose=True,
                              input_names=[ "input" ] , output_names=['output'],
                              opset_version = opset_version,
                              operator_export_type=operator_export_type)
            onnx_model = onnx.load(onnx_file)
            onnx.checker.check_model(onnx_model)
        else:
            print("[ERR]:",e)
    except Exception as e:
        print("[ERR]:", e)

def infer_onnx(onnx_file, input_data, benchmark=False):
    ort_session = onnxruntime.InferenceSession(onnx_file)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    if benchmark:
        ort_outs = bm(ort_session.run)(None, ort_inputs)
    else:
        ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]
