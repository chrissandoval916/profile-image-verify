import tensorflow as tf
import tensorflow_hub as hub

FILE_NAME = ''
FAKE_QUANT_OPS = []


def create_module_graph(module_spec):
    """
    Creates a graph and loads Hub Module into it
    :param module_spec: the hub.ModuleSpec for the image module being used
    :return:
        graph: the tf.Graph that was created
        bottleneck_tensor: bottleneck values output by the module
        resized_input_tensor: input images, resized as expected by the module
        wants_quantization: boolean if the module has been instrumented with fake quantization ops
    """
    height, width = hub.get_expected_image_size(module_spec)
    with tf.Graph().as_default() as graph:
        resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
        m = hub.Module(module_spec)
        bottleneck_tensor = m(resized_input_tensor)
        wants_quantization = any(node.op in FAKE_QUANT_OPS
                                 for node in graph.as_graph_def().node)
    return graph, bottleneck_tensor, resized_input_tensor, wants_quantization


def save_graph_to_file(graph_file_name, sess):
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FILE_NAME]
    )

    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
