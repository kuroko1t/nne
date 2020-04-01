import onnx
import torch
import tensorflow as tf
from nne.common import *
if check_jetson():
    from torch2trt import torch2trt, TRTModule
else:
    import onnxruntime
    import onnx_tf
    from onnx_tf.backend import prepare
import os
import re
from nne.benchmark import benchmark as bm

def check_model_is_cuda(model):
    return next(model.parameters()).is_cuda

def cv2tflite(model, input_shape, tflite_path, edgetpu=False):
    """
    convert torch model to tflite model using onnx
    """
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')
    tmp_onnx_path = 'tmp.onnx'
    tmp_pb_path = 'tmp.pb'
    torch.onnx.export(model, dummy_input, tmp_onnx_path,
                      do_constant_folding=True,
                      input_names=[ "input" ] , output_names=['output'])

    onnx_model = onnx.load('./tmp.onnx')
    onnx_input_names = [input.name for input in onnx_model.graph.input]
    onnx_output_names = [output.name for output in onnx_model.graph.output]

    os.system(f'onnx-tf convert -i {tmp_onnx_path} -o {tmp_pb_path}')

    input_data = ''
    if dummy_input.is_cuda:
        input_data = dummy_input.cpu().numpy()
    else:
        input_data = dummy_input.numpy()
    train = tf.convert_to_tensor(input_data)
    my_ds = tf.data.Dataset.from_tensor_slices((train)).batch(10)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        tmp_pb_path,
        onnx_input_names,
        onnx_output_names
    )
    if edgetpu:
        def representative_dataset_gen():
            for input_value in my_ds.take(10):
                yield [input_value]
        converter.representative_dataset = representative_dataset_gen
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    try:
        tflite_model = converter.convert()
    except Exception as e:
        if 'you will need custom implementations' in e.args[-1]:
            converter.allow_custom_ops = True
            tflite_model = converter.convert()
        else:
            print('[ERR]:', e)

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    os.remove(tmp_onnx_path)
    os.remove(tmp_pb_path)

    if edgetpu:
        os.system(f'edgetpu_compiler {tflite_path}')

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

def cv2trt(model, input_shape, trt_file, fp16_mode=False):
    """
    convert torch model to tflite model using onnx
    """
    model.eval()
    if check_model_is_cuda(model):
        dummy_input = torch.randn(input_shape, device='cuda')
    else:
        dummy_input = torch.randn(input_shape, device='cpu')
    model_trt = torch2trt(model, [dummy_input], fp16_mode=fp16_mode, max_batch_size=input_shape[0])
    torch.save(model_trt.state_dict(), trt_file)

def infer_onnx(onnx_file, input_data, benchmark=False):
    ort_session = onnxruntime.InferenceSession(onnx_file)
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    if benchmark:
        ort_outs = bm(ort_session.run)(None, ort_inputs)
    else:
        ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

def infer_trt(trt_file, input_data, benchmark=False):
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(trt_file))
    input_data = torch.from_numpy(input_data).cuda()
    if benchmark:
        output = bm(model_trt)(input_data)
    else:
        output = model_trt(input_data)
    return output.detach().cpu().numpy()
    
def infer_tflite(tflitepath, input_data, benchmark=False):
    interpreter = tf.lite.Interpreter(model_path=tflitepath)
    # allocate memory
    interpreter.allocate_tensors()

    # get model input and output propaty
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ## get input shape
    #input_shape = input_details[0]['shape']

    def execute():
        # set tensor pointer to index
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # execute infer
        interpreter.invoke()
        # get result from index of output_details
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data
    if benchmark:
        output_data = bm(execute)()
    else:
        output_data = execute()
    return output_data

def infer_torch(model, input_data,  benchmark=False):
    """
    model : loaded model
    input_data: numpy array
    """
    model.eval()
    input_data = torch.from_numpy(input_data)
    if check_model_is_cuda:
        input_data = input_data.cuda()
    if benchmark:
        output = bm(model)(input_data)
    else:
        output = model(input_data)
    return output.detach().cpu().numpy()
