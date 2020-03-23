import torch
import onnx
import onnx_tf
from onnx_tf.backend import prepare
import tensorflow as tf
import os

def convert2tflite(model, dummy_input, tflite_path):
    """
    convert torch model to tflite model using onnx
    """
    tmp_onnx_path = 'tmp.onnx'
    tmp_pb_path = 'tmp.pb'
    torch.onnx.export(model, dummy_input, tmp_onnx_path, verbose=True,
                      input_names=[ "dummy_inputs" ] , output_names=['dummy_outputs'])

    onnx_model = onnx.load('./tmp.onnx')
    onnx_input_names = [input.name for input in onnx_model.graph.input]
    onnx_output_names = [output.name for output in onnx_model.graph.output]
    print('kkkk--',onnx_input_names)

    os.system(f'onnx-tf convert -i {tmp_onnx_path} -o {tmp_pb_path}')

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        tmp_pb_path,
        onnx_input_names,
        onnx_output_names
    )
    
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

