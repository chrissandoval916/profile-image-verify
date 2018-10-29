import tensorflow as tf

from library.tensorflow.configuration import get_final_retrain_ops
from library.graph.tensorflow import create_module_graph, save_graph_to_file
from library.bottleneck import get_random_cached_bottlenecks

TENSOR_NAME = ''
CHECKPOINT_NAME = ''
TFHUB_MODULE = ''
PRINT_MISCLASSIFIED_IMAGES = False
TEST_BATCH_SIZE = 0
BOTTLENECK_DIR = ''
IMAGE_DIR = ''


def save_graph(graph, graph_file_name, module_spec, class_count):
    """Saves a graph to file, creating a valid quantized one if necessary"""
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    save_graph_to_file(graph_file_name, sess)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """
    Inserts the operations we need to evaluate the accuracy of our results
    """
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(prediction, ground_truth_tensor)
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def build_eval_session(module_spec, class_count):
    """
    Builds a restored eval session without train operations for exporting
    :param module_spec: The hub.ModuleSpec for the image module being used
    :param class_count: Number of classes
    :return: Eval session containing the retored eval graph: bottleneck input,
             ground truth, eval step, and prediction tensors
    """
    # If quantized, we need to create the correct eval graph for exporting
    eval_graph, bottleneck_tensor, resized_input_tensor, wants_quantization = (create_module_graph(module_spec))

    eval_sess = tf.Session(graph=eval_graph)
    with eval_graph.as_default():
        # Add the new layer for exporting
        (_, _, bottleneck_input, ground_truth_input, final_tensor) = get_final_retrain_ops(
            class_count, TENSOR_NAME, bottleneck_tensor, wants_quantization, is_training=False
        )

        # Now we need to restore the values from the training graph to the eval graph
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction)


def run_final_eval(train_session, module_spec, class_count, image_lists, jpeg_data_tensor,
                   decoded_image_tensor, resized_image_tensor, bottleneck_tensor):
    """
    Run a final evaluation on an eval graph using the test data set
    :param train_session: Session for the train graph with the tensors below
    :param module_spec: The hub.ModuleSpec for the image module being used
    :param class_count: Number of classes
    :param image_lists: OrderedDict of training images for each label
    :param jpeg_data_tensor: Layer to feed jpeg image data into
    :param decoded_image_tensor: Output of decoding and resizing the image
    :param resized_image_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Bottleneck output layer of the CNN graph
    :return:
    """
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            train_session, image_lists, TEST_BATCH_SIZE,
            'testing', BOTTLENECK_DIR, IMAGE_DIR, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor, TFHUB_MODULE
        )
    )
    (eval_session, _, bottleneck_input, ground_truth_input, evaluation_step, prediction) = build_eval_session(
        module_spec, class_count
    )
    test_accuracy, predictions = eval_session.run(
        [evaluation_step, prediction],
        feed_dict={
            bottleneck_input: test_bottlenecks,
            ground_truth_input: test_ground_truth
        }
    )
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks))
    )

    if PRINT_MISCLASSIFIED_IMAGES:
        tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i]:
                tf.logging.info('%70s %s' % (test_filename, list(image_lists.keys())[predictions[i]]))


def export_model(module_spec, class_count, saved_model_dir):
    """
    Exports model for serving
    :param module_spec: The hub.ModuleSpec for the image module being used.
    :param class_count: The number of classes.
    :param saved_model_dir: irectory in which to save exported model and variables.
    :return:
    """
    # SavedModel should hold the eval graph
    sess, in_image, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph
    with graph.as_default():
        inputs = {'image': tf.saved_model.utils.build_tensor_info(in_image)}

        out_classes = sess.graph.get_tensor_by_name('profile_image_verifier:0')
        outputs = {
            'prediction': tf.saved_model.utils.build_tensor_info(out_classes)
        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Save out the SavedModel
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            },
            legacy_init_op=legacy_init_op
        )
        builder.save()