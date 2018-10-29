from library.graph.interface import GraphInterface
import tensorflow as tf
import tensorflow_hub as hub


class TensorflowGraph(GraphInterface):

    FAKE_QUANT_OPS = []
    CLASSIFICATION_LAYER_NAME = ''
    OUTPUT_GRAPH_FILE_PATH = ''
    INTERMEDIATE_GRAPH_FILE_DIR = ''
    GRAPH_OUTPUT = {}

    def ___init___(self, classification_layer_name, fake_quant_ops, output_graph_file_path, intermediate_graph_file_dir):
        self.CLASSIFICATION_LAYER_NAME = classification_layer_name
        self.FAKE_QUANT_OPS = fake_quant_ops
        self.OUTPUT_GRAPH_FILE_PATH = output_graph_file_path
        self.INTERMEDIATE_GRAPH_FILE_DIR = intermediate_graph_file_dir

        self.set_graph_name(self.OUTPUT_GRAPH_FILE_PATH)

    def create(self, module_spec):
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
            wants_quantization = any(node.op in self.FAKE_QUANT_OPS
                                     for node in graph.as_graph_def().node)
        return graph, bottleneck_tensor, resized_input_tensor, wants_quantization

    def save(self):
        with tf.gfile.FastGFile(self.GRAPH_NAME, 'wb') as f:
            f.write(self.GRAPH_OUTPUT.SerializeToString())

    def set_graph_output(self, session):
        graph = session.graph

        self.GRAPH_OUTPUT = tf.graph_util.convert_variables_to_constants(
            session, graph.as_graph_def(), [self.CLASSIFICATION_LAYER_NAME]
        )

    def save_intermediate_graph(self, session, step):
        intermediate_file_name = (self.INTERMEDIATE_GRAPH_FILE_DIR + 'intermediate_' + str(step) + '.pb')

        tf.logging.info('Save intermediate result to: ' + intermediate_file_name)
        self.set_graph_name(intermediate_file_name)
        self.set_graph_output(session)
        self.save()

    def save_output_graph(self, session):
        tf.logging.info('Save final result to : ' + self.OUTPUT_GRAPH_FILE_PATH)
        self.set_graph_name(self.OUTPUT_GRAPH_FILE_PATH)
        self.set_graph_output(session)
        self.save()
