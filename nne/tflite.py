import onnx
import torch
import tensorflow as tf
import os
import sys
from .common import *

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
            tflite_model = converter.convert()
        else:
            print('[ERR]:', e)
            sys.exit()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    os.remove(tmp_onnx_path)
    os.remove(tmp_pb_path)

    if edgetpu:
        os.system(f'edgetpu_compiler {tflite_path}')


def load_tflite(tflitepath):
    interpreter = tf.lite.Interpreter(model_path=tflitepath)
    # allocate memory
    interpreter.allocate_tensors()
    return interpreter


def infer_tflite(interpreter, input_data, bm=None):
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
    if bm:
        output_data = bm.measure(execute, name='tflite')()
    else:
        output_data = execute()
    return output_data
