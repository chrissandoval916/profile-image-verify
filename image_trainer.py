from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import sys
import tensorflow as tf
import tensorflow_hub as hub

import library.tensorflow.configuration as tc
import library.tensorflow.execution as te
import library.image.distortions as distortion
import library.image.jpeg as jpeg
import library.graph as grapher
import library.bottleneck as bottleneck

ARGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

IMAGE_EXT = ['jpg', 'jpeg', 'JPG', 'JPEG']

# The location where variable checkpoints will be stored
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify images using Tensorflow API')
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folder of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='./tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--intermediate_output_graphs_dir',
        type=str,
        default='./tmp/intermediate_graph/',
        help='Where to save the intermediate graphs.'
    )
    parser.add_argument(
        '--intermediate_store_frequency',
        type=int,
        default=0,
        help="""\
             How many steps to store intermediate graph. If "0" then will not
             store.\
          """
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='./tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='./tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\
          How many images to test on. This test set is only used once, to evaluate
          the final accuracy of the model after training completes.
          A value of -1 causes the entire test set to be used, which leads to more
          stable results across runs.\
          """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\
          How many images to use in an evaluation batch. This validation set is
          used much more often than the test set, and is an early indicator of how
          accurate the model is during training.
          A value of -1 causes the entire validation set to be used, which leads to
          more stable results across training iterations, but may be slower on large
          training sets.\
          """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\
          Whether to print out a list of all misclassified test images.\
          """,
        action='store_true'
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='./tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='profile_image_verifier',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\
          Whether to randomly flip half of the training images horizontally.\
          """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\
          A percentage determining how much of a margin to randomly crop off the
          training images.\
          """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\
          A percentage determining how much to randomly scale up the size of the
          training images by.\
          """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\
          A percentage determining how much to randomly multiply the training image
          input pixels up or down by.\
          """
    )
    parser.add_argument(
        '--tfhub_module',
        type=str,
        default=(
            'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
        help="""\
          Which TensorFlow Hub module to use.
          See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
          for some publicly available ones.\
          """
    )
    parser.add_argument(
        '--saved_model_dir',
        type=str,
        default='./saved_model',
        help='Where to save the exported graph.'
    )
    return parser


def set_module_params():
    grapher.FILE_NAME = ARGS.final_tensor_name
    grapher.FAKE_QUANT_OPS = FAKE_QUANT_OPS
    tc.IMAGE_EXT = IMAGE_EXT
    tc.MAX_NUM_IMAGES_PER_CLASS = MAX_NUM_IMAGES_PER_CLASS
    tc.SUMMARIES_DIR = ARGS.summaries_dir
    tc.GRAPH_DIR = ARGS.intermediate_output_graphs_dir
    tc.STORE_FREQUENCY = ARGS.intermediate_store_frequency
    te.TENSOR_NAME = ARGS.final_tensor_name
    te.CHECKPOINT_NAME = CHECKPOINT_NAME
    te.TFHUB_MODULE = ARGS.tfhub_module
    te.PRINT_MISCLASSIFIED_IMAGES = ARGS.print_misclassified_test_images
    te.TEST_BATCH_SIZE = ARGS.test_batch_size
    te.BOTTLENECK_DIR = ARGS.bottleneck_dir
    te.IMAGE_DIR = ARGS.image_dir
    bottleneck.MAX_NUM_IMAGES_PER_CLASS = MAX_NUM_IMAGES_PER_CLASS


def main_func(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not ARGS.image_dir:
        tf.logging.error('Must set flag --image_dir.')
        return -1

    # Prepare necessary directories that can be used during training
    tc.prepare_tensorboard_dirs()

    # Look at the img dir structure and create lists of all images
    image_lists = tc.create_image_lists(ARGS.image_dir, ARGS.testing_percentage,
                                     ARGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error("No valid folders of images found at '" + ARGS.image_dir + "'")
        return -1
    if class_count == 1:
        tf.logging.error("Only one valid folder of images found at '" +
                         ARGS.image_dir + "' - multiple classes are needed for classification")
        return -1

    # Check for image distortion command-line flags
    do_distort_images = distortion.should_distort_images(
        ARGS.flip_left_right, ARGS.random_crop, ARGS.random_scale, ARGS.random_brightness
    )

    # Set up the pre-trained graph
    module_spec = hub.load_module_spec(ARGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (grapher.create_module_graph(module_spec))

    # Add the new layer that we'll be training
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = tc.get_final_retrain_ops(
            class_count, ARGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training=True
        )

    with tf.Session(graph=graph) as sess:
        # Initialize all weights for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph
        jpeg_data_tensor, decoded_image_tensor = jpeg.add_jpeg_decoding(module_spec)

        if do_distort_images:
            # Image distortion has been enabled so set up distortion operations
            (distorted_jpeg_data_tensor, distorted_image_tensor) = distortion.add_input_distortions(
                ARGS.flip_left_right, ARGS.random_crop, ARGS.random_scale, ARGS.random_brightness, module_spec
            )
        else:
            # Calculate bottleneck image summaries and cache them on disk
            bottleneck.cache_bottlenecks(sess, image_lists, ARGS.image_dir,
                              ARGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, ARGS.tfhub_module)

        # Create the operations we need to evaluate the accuracy of our new layer
        evaluation_step, _ = te.add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(ARGS.summaries_dir + '/train', sess.graph)

        validation_writer = tf.summary.FileWriter(ARGS.summaries_dir + '/validation')

        # Create a train saver that is used to restore values into an eval graph when exporting models
        train_saver = tf.train.Saver()

        # Run the training for as many cycles as requested on the command line
        for i in range(ARGS.how_many_training_steps):
            # Get a batch of input bottleneck values, either calculated fresh due to applied distortion,
            # or from the cache stored in the bottleneck dir
            if do_distort_images:
                (train_bottlenecks, train_ground_truth) = bottleneck.get_random_distorted_bottlenecks(
                    sess, image_lists, ARGS.train_batch_size, 'training',
                    ARGS.image_dir, distorted_jpeg_data_tensor, distorted_image_tensor,
                    resized_image_tensor, bottleneck_tensor
                )
            else:
                (train_bottlenecks, train_ground_truth, _) = bottleneck.get_random_cached_bottlenecks(
                    sess, image_lists, ARGS.train_batch_size, 'training',
                    ARGS.bottleneck_dir, ARGS.image_dir, jpeg_data_tensor,
                    decoded_image_tensor, resized_image_tensor, bottleneck_tensor, ARGS.tfhub_module
                )
            # Feed the bottlenecks and ground truth into the graph, and run a training step
            # Capture training summaries for TensorBoard with the 'merged' op
            train_summary, _ = sess.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth}
            )
            train_writer.add_summary(train_summary, i)

            # Every so often print out how well the graph is training
            is_last_step = (i + 1 == ARGS.how_many_training_steps)
            if (i % ARGS.eval_step_interval) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = sess.run(
                    [evaluation_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth}
                )
                tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                (datetime.now(), i, train_accuracy * 100))
                tf.logging.info('%s: Step %d: Cross entropy = %f' %
                               (datetime.now(), i, cross_entropy_value))
                # TODO: Make this use an eval graph, to avoid quantization
                # moving averages being updated by the validation set, though in
                # practice this makes a negligible difference.
                validation_bottlenecks, validation_ground_truth, _ = (
                    bottleneck.get_random_cached_bottlenecks(
                        sess, image_lists, ARGS.validation_batch_size, 'validation',
                        ARGS.bottleneck_dir, ARGS.image_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor, bottleneck_tensor, ARGS.tfhub_module
                    )
                )
                # Run a validation step and capture training summaries for TensorBoard with the 'merged' op
                validation_summary, validation_accuracy = sess.run(
                    [merged, evaluation_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth}
                )
                validation_writer.add_summary(validation_summary, i)
                tf.logging.info('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                                (datetime.now(), i, validation_accuracy * 100, len(validation_bottlenecks))
                )

                # Store intermediate results
                intermediate_frequency = ARGS.intermediate_store_frequency

                if intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0:
                    # If we want to do an intermediate save, save a checkpoint of the train graph
                    # to restore into the eval graph
                    train_saver.save(sess, CHECKPOINT_NAME)
                    intermediate_file_name = (ARGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to: ' + intermediate_file_name)
                    te.save_graph(graph, intermediate_file_name, module_spec, class_count)

        # After training is complete, force one last save of the train checkpoint
        train_saver.save(sess, CHECKPOINT_NAME)

        # Training complete, so run a final test evaluation on some new images we haven't used before
        te.run_final_eval(sess, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor,
                       resized_image_tensor, bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as constants
        tf.logging.info('Save final result to : ' + ARGS.output_graph)
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')
        te.save_graph(graph, ARGS.output_graph, module_spec, class_count)
        with tf.gfile.FastGFile(ARGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if ARGS.saved_model_dir:
            te.export_model(module_spec, class_count, ARGS.saved_model_dir)


if __name__ == '__main__':

    # args - the input params needed for this script
    # unparsed_args - extra args or those needed for the tensorflow app call
    ARGS, unparsed_args = build_arg_parser().parse_known_args()
    set_module_params()
    tf.app.run(main=main_func, argv=[sys.argv[0]] + unparsed_args)
