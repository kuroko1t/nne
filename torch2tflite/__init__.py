import torch
import onnx
import onnx_tf
from onnx_tf.backend import prepare
import tensorflow as tf
import os

#tf.enable_eager_execution()

def convert2tflite(model, dummy_input, tflite_path, edgetpu=False):
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

    os.system(f'onnx-tf convert -i {tmp_onnx_path} -o {tmp_pb_path}')

    train = tf.convert_to_tensor(dummy_input.numpy())
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
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

def infer_tflite(tflitepath, input_data):
    interpreter = tf.lite.Interpreter(model_path="alexnet.tflite")
    # allocate memory
    interpreter.allocate_tensors()

    # get model input and output propaty
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ## get input shape
    #input_shape = input_details[0]['shape']

    # set tensor pointer to index
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # execute infer
    interpreter.invoke()
    # get result from index of output_details
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data
