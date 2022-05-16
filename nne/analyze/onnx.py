import re
import collections
import onnx
import json
from onnx import defs


class Analyze:
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        model = onnx.load(model_path)
        graph_def = model.graph
        self.input_shapes = self.get_input_shape(graph_def)
        self.output_shapes = self.get_output_shape(graph_def)
        self.opset_info = model.opset_import[0].version
        self.nodes = self.nodes(graph_def)
        self.model_info = self.modelinfo()
        if output_path:
            self.dump(self.model_info, output_path)

    def modelinfo(self):
        model_info = {}
        model_info["OpsetVersion"] = self.opset_info
        model_info["InputShape"] = self.input_shapes
        model_info["OutputShape"] = self.output_shapes
        node_dict_list = [model_info] + self.nodes2dict()
        return node_dict_list

    def nodes(self, graph_def):
        return [OnnxNode(node) for node in graph_def.node]

    def get_input_shape(self, graph_def):
        input_shapes = []
        if graph_def.initializer:
            initialized = {init.name for init in graph_def.initializer}
        else:
            initialized = set()
        for value_info in graph_def.input:
            if value_info.name in initialized:
                continue
            shape = list(
                d.dim_value if (
                    d.dim_value > 0 and d.dim_param == "") else None
                for d in value_info.type.tensor_type.shape.dim)
            input_shapes.append(shape)
        return input_shapes

    def get_output_shape(self, graph_def):
        output_shapes = []
        for value_info in graph_def.output:
            output_shape = [
                dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            output_shapes.append(output_shape)
        return output_shapes

    def unique_nodes(self):
        # [re.sub("_\d+", '', node.name) for node in self.nodes]
        strip_node_names = [node.op_type for node in self.nodes]
        counter_nodes = collections.Counter(strip_node_names)
        return dict(counter_nodes)

    def nodes2dict(self):
        node_dict_list = []
        for node in self.nodes:
            node_dict = collections.OrderedDict()
            node_dict["name"] = node.name
            node_dict["op_type"] = node.op_type
            if node.attrs:
                if "value" in node.attrs:
                    node_attrs = node.attrs.pop("value")
                else:
                    node_attrs = node.attrs
                node_dict["attrs"] = node.attrs
                node_dict_list.append(node_dict)
        return node_dict_list

    def dump(self, node_dict_list, output_path):
        with open(output_path, "w") as f:
            json.dump(node_dict_list, f, indent=2)

    def summary(self):
        print()
        print("#### SUMMARY ONNX MODEL ####")
        print("opset:", self.opset_info)
        print("INPUT:", self.input_shapes)
        print("OUTPUT:", self.output_shapes)
        unique_nodes_count = self.unique_nodes()
        print(f"--Node List-- num({len(self.nodes)})")
        print(unique_nodes_count)


class OnnxNode(object):
    def __init__(self, node):
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            translate_onnx(
                                attr.name, convert_onnx_attribute_proto(attr)))
                           for attr in node.attribute])
        self.inputs = list(node.input)
        self.outputs = list(node.output)
        self.node_proto = node


def analyze_graph(model_path, output_path):
    analyzer = Analyze(model_path, output_path)
    analyzer.summary()
    return analyzer.model_info


def translate_onnx(key, val):
    return onnx_attr_translator.get(key, lambda x: x)(val)


onnx_attr_translator = {
    "axis": lambda x: int(x),
    "axes": lambda x: [int(a) for a in x],
    "dtype": lambda x: x,
    "keepdims": lambda x: bool(x),
    "to": lambda x: x,
}


def convert_onnx_attribute_proto(attr_proto):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.
    NB: Tensor attribute gets returned as the straight proto.
    """
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.HasField('g'):
        return attr_proto.g
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        str_list = list(map(lambda x: str(x, 'utf-8'), str_list))
        return str_list
    elif attr_proto.HasField('sparse_tensor'):
        return attr_proto.sparse_tensor
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))
