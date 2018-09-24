from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

ARGS = None

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

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


def ensure_dir_exists(dir_name):
    """
    Make sure the folder exists on disk
    :param dir_name: path string to the folder we want to create
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_image_path(image_lists, label_name, index, image_dir, category):
    """
    Return a path to an image for a label at the given index
    :param image_lists: OrderedDict of training images for each label
    :param label_name: Label string we want to get an image for
    :param index: Integer offset of the image we want. This will be modulo-ed by the
                  available number of images for the label, so it can be arbitrarily large
    :param image_dir: Root folder string containing all sub dirs of training images
    :param category: Name string of which set to pull images from (training, testing, validation)
    :return:
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s', label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index,
                        bottleneck_dir, category, module_name):
    """
    Return a path to a bottleneck file for a label at the given index
    :param image_lists: OrderedDict of training images for each label
    :param label_name: Label string we want to get an image for
    :param index: Integer offset of the image we want. This will be modulo-ed by the
                  available number of images for the label, so it can be arbitrarily large
    :param bottleneck_dir: Folder string that will hold cached files of bottleneck values
    :param category: Name string of which set to pull images from (training, testing, validation)
    :param module_name: Name of the image module being used
    :return: File system path string to an image that meets the requested parameters
    """
    module_name = (module_name.replace('://', '~')  # URL scheme.
                   .replace('/', '~')  # URL and Unix paths.
                   .replace(':', '~').replace('\\', '~'))  # Windows paths.
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '_' + module_name + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor, bottleneck_tensor):
    """
    Runs inference on an image to extract the bottleneck summary layer
    :param sess: Current active TensorFlow Session
    :param image_data: String of raw JPEG data
    :param image_data_tensor: Input data layer in the graph
    :param decoded_image_tensor: Output of initial image resizing and preprocessing
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Lyer before the final softmax
    :return: Numpy array of bottleneck values
    """
    # First decode the JPEG image, resize it, and rescale the pixel values
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # Then run it through the recognition network
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
    """
    Create a single bottleneck file
    :param bottleneck_path: File system path string to bottleneck file
    :param image_lists: OrderedDict of training images for each label
    :param label_name: Label string we want to get an image for
    :param index: Integer offset of the image we want. This will be modulo-ed by the
                  available number of images for the label, so it can be arbitrarily large
    :param image_dir: Root folder string containing all sub dirs of training images
    :param category: Name string of which set to pull images from (training, testing, validation)
    :param sess: Current active TensorFlow Session
    :param jpeg_data_tensor: Input tensor for jpeg data from file
    :param decoded_image_tensor: Output of decoding and resizing the image
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Penultimate output layer of the graph
    :return:
    """
    tf.logging.info('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    try:
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor,
                                                    decoded_image_tensor, resized_input_tensor, bottleneck_tensor)
    except Exception as e:
        raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, module_name):
    """
    If a cached version of the bottleneck data exists, return it. Else generate and save
    :param sess: Current active TensorFlow Session
    :param image_lists: OrderedDict of training images for each label
    :param label_name: Label string we want to get an image for
    :param index: Integer offset of the image we want. This will be modulo-ed by the
                  available number of images for the label, so it can be arbitrarily large
    :param image_dir: Root folder string containing all sub dirs of training images
    :param category: Name string of which set to pull images from (training, testing, validation)
    :param bottleneck_dir: Folder string that will hold cached files of bottleneck values
    :param jpeg_data_tensor: Input tensor for jpeg data from file
    :param decoded_image_tensor: Output of decoding and resizing the image
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Penultimate output layer of the graph
    :param module_name: Name of the image module being used
    :return: Numpy array of values produced by the bottleneck layer for the image
    """
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          bottleneck_dir, category, module_name)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except ValueError:
        tf.logging.warning('Invalid float found, recreating bottleneck')
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                               image_dir, category, sess, jpeg_data_tensor,
                               decoded_image_tensor, resized_input_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # Dont worry about exceptions as they shouldn't happen after a fresh creation
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, module_name):
    """
    If there are no distortions applied, we are very likely to read the same image
    multiple times. As such we want to only calculate the bottleneck layer values once
    for each image durin preprocessing and then just read the cached values repeatedly
    during training. This function goes through all the images, calculates the values,
    and then caches them
    :param sess: Current active TensorFlow Session
    :param image_lists: OrderedDict of training images for each label
    :param image_dir: Root folder string containing all sub dirs of training images
    :param bottleneck_dir: Folder string that will hold cached files of bottleneck values
    :param jpeg_data_tensor: Input tensor for jpeg data from file
    :param decoded_image_tensor: Output of decoding and resizing the image
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Penultimate output layer of the graph
    :param module_name: Name of the image module being used
    :return:
    """
    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                                         category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                         resized_input_tensor, bottleneck_tensor, module_name)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    tf.logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')


def prepare_tensorboard_dirs():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(ARGS.summaries_dir):
        tf.gfile.DeleteRecursively(ARGS.summaries_dir)
    tf.gfile.MakeDirs(ARGS.summaries_dir)
    if ARGS.intermediate_store_frequency > 0:
        ensure_dir_exists(ARGS.intermediate_output_graphs_dir)
    return


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Build a list of training images from the passed directory subdirectories

    We will be skipping the root dir and focusing on the sub dirs. The following
    tensorflow calls will analyze the sub dirs and split them into stable training,
    testing, and validation sets. The return will be a data structure describing the
    lists of images for each lable and their paths.

    :param image_dir: String path to a folder containing sub dirs of images
    :param testing_percentage: Integer percentage of the images to reserve for tests
    :param validation_percentage: Integer percentage of images reserved for validation
    :return: OrderedDict containing an entry for each label sub dir with an object that
             contains the following key/value pairs:
             'dir': dir name
             'training': array of training image file names
             'testing': array of training image file names
             'validation': array of training image file names
    """
    if not tf.gfile.Exists(image_dir):
        tf.logging.error("Attempting to create list of training images but '" + image_dir + "' not found.")
        return None
    result = collections.OrderedDict()
    sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
    # Skip the root dir, we only want the sub dirs
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning("No files found in {}. Make sure extension is one of the following: {}".format(dir_name, extensions))
            continue
        if len(file_list) < 20:
            tf.logging.warning("WARNING: '" + dir_name + "' dir has less than 20 image (100 recommended), which may cause issues.")
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('WARNING {} dir has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # Ignore anything after '_nohash_' in the file name when deciding
            # which set to put an image in, the data set creator has a way of
            # grouping photos that are close variations of each other.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # We need to decide whether this file should go into the training,
            # testing, or validation sets, and we want to keep existing files in
            # the same set even if morefiles are subsequently added. To do that,
            # we need a stable way of deciding based on just the file name itself,
            # so we do a hash of that and then use that to generate a probability
            # value that we use to assign it
            hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def should_distort_images(flip_left_right, random_crop, random_scale, random_brightness):
    """
    Check if any distortions have been enabled from command flags

    :param flip_left_right: boolean to randomly mirror images horizontally
    :param random_crop: integer percentage setting the total margin used around the crop box
    :param random_scale: integer percentage of how much to vary the scale by
    :param random_brightness: integer range to randomly multiply the pixel values by
    :return: boolean value indication if any distortions should be applied
    """
    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale, random_brightness, module_spec):
    """
    Creates the operations to apply the specified distortions.

    During training it can help to improve the results if we run the images
    through simple distortions like crops, scales, and flips. These reflect the
    kind of variations we expect in the real world, and so can help train the
    model to cope with natural data more effectively. Here we take the supplied
    parameters and construct a network of operations to apply them to an image.

    Cropping
    ~~~~~~~~

    Cropping is done by placing a bounding box at a random position in the full
    image. The cropping parameter controls the size of that box relative to the
    input image. If it's zero, then the box is the same size as the input and no
    cropping is performed. If the value is 50%, then the crop box will be half the
    width and height of the input. In a diagram it looks like this:

    <       width         >
    +---------------------+
    |                     |
    |   width - crop%     |
    |    <      >         |
    |    +------+         |
    |    |      |         |
    |    |      |         |
    |    |      |         |
    |    +------+         |
    |                     |
    |                     |
    +---------------------+

    Scaling
    ~~~~~~~

    Scaling is a lot like cropping, except that the bounding box is always
    centered and its size varies randomly within the given range. For example if
    the scale percentage is zero, then the bounding box is the same size as the
    input and no scaling is applied. If it's 50%, then the bounding box will be in
    a random range between half the width and height and full size.
    :param flip_left_right: Boolean whether to randomly mirror images horizontally
    :param random_crop: Integer percentage setting the total margin used around the
                        crop box
    :param random_scale: Integer percentage of how much to vary the scale by
    :param random_brightness: Integer range to randomly multiply the pixel values by graph
    :param module_spec: The hub.ModuleSpec for the image module being used
    :return:
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range[0,1] of float32
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(shape=[], minval=1.0, maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, input_width)
    precrop_height = tf.multiply(scale_value, input_height)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d, precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d, [input_height, input_width, input_depth])

    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(shape=[], minval=brightness_min, maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


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


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization)
    :param var: incoming summaries
    :return:
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
                           quantize_layer, is_training):
    batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()
    assert batch_size is None, 'We want to work with arbitrary batch size.'
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[batch_size, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder'
        )

        ground_truth_input = tf.placeholder(
            tf.int64, [batch_size], name='GroundTruthInput'
        )

    # Organizing the following ops so they are easier to see in TensorBoard
    layer_name = 'final_retrain_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal(
                [bottleneck_tensor_size, class_count], stddev=0.001
            )
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)

    # The tf.contrib.quantize functions rewrite the graph in place for quantization.
    # The imported model graph has already been rewritten, so upon calling these rewrites,
    # only the newly added final layer will be transformed
    if quantize_layer:
        if is_training:
            tf.contrib.quantize.create_training_graph()
        else:
            tf.contrib.quantize.create_eval_graph()

    tf.summary.histogram('activations', final_tensor)

    # If this is an eval graph, we dont need to add loss ops or an optimizer
    if not is_training:
        return None, None, bottleneck_input, ground_truth_input, final_tensor

    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
            labels=ground_truth_input, logits=logits
        )

    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(ARGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


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


def add_jpeg_decoding(module_spec):
    """
    Operations that perform JPEG decoding and resizing to the graph
    :param module_spec: hub.ModuleSpec for the image module being used
    :return: Tensors for the node to feed JPEG data into, and the ouput of
             the preprocessing steps
    """
    input_height, input_width = hub.get_expected_image_size(module_spec)
    input_depth = hub.get_num_image_channels(module_spec)
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
    # Convert from full range of uint8 to range [0,1] of float32
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image, tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)

    return jpeg_data, resized_image


def get_random_distorted_bottlenecks(sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
                                     distorted_image, resized_input_tensor, bottleneck_tensor):
    """
    If training with distortion (crops,scales,flips,etc) we have to recalculate the full model
    for every image, and so we cant used cached bottleneck values. Instead we find random images
    for the requested category, run them through the distortion graph, and then the full graph
    to get the bottleneck results for each
    :param sess: Current TensorFlow Session
    :param image_lists: OrderedDict of training images for each label
    :param how_many: Integer number for the number of bottleneck values to return
    :param category: Name string of which set to pull images from (training, testing, validation)
    :param image_dir: Root folder string containing all sub dirs of training images
    :param input_jpeg_tensor: Input layer we feed the image data to
    :param distorted_image: Output node of the distortion graph
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Bottleneck output layer of the CNN graph
    :return: List of bottleneck arrays and their corresponding ground truths
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir, category)

        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        jpeg_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # TODO possible optimization point
        # Note that we materialize the distorted_image_data as a numpy array
        # before sending running inference on the image. This involves 2 memory
        # copies and might be optimized in other implements
        distorted_image_data = sess.run(distorted_image, {input_jpeg_tensor: jpeg_data})
        bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
        bottleneck_values = np.squeeze(bottleneck_values)
        bottlenecks.append(bottleneck_values)
        ground_truths.append(label_index)
    return bottlenecks, ground_truths


def get_random_cached_bottlenecks(sess, image_lists, how_many, category, bottleneck_dir, image_dir,
                                  jpeg_data_tensor, decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, module_name):
    """
    If no distortions are being applied, use the cached bottleneck values directly from disk. Picks
    a random set of images from the specified category
    :param sess: Current TensorFlow Session
    :param image_lists: OrderedDict of training images for each label
    :param how_many: Integer number for the number of bottleneck values to return
    :param category: Name string of which set to pull images from (training, testing, validation)
    :param bottleneck_dir: Folder string that contains cached bottleneck value files
    :param image_dir: Root folder string containing all sub dirs of training images
    :param jpeg_data_tensor: Layer to feed teh image data into
    :param decoded_image_tensor: Output of decoding and resizing the image
    :param resized_input_tensor: Input node of the recognition graph
    :param bottleneck_tensor: Bottleneck output layer of the CNN graph
    :param module_name: name of the image module being used
    :return: List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
    """
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        # Retrieve a random sample of bottlenecks
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir,
                                                  category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                                  resized_input_tensor, bottleneck_tensor, module_name)
            bottlenecks.append(bottleneck)
            ground_truths.append(label_index)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index, image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index, image_dir,
                                                      category, bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
                                                      resized_input_tensor, bottleneck_tensor, module_name)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


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
        (_, _, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, ARGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training=False
        )

        # Now we need to restore the values from the training graph to the eval graph
        tf.train.Saver().restore(eval_sess, CHECKPOINT_NAME)

        evaluation_step, prediction = add_evaluation_step(final_tensor, ground_truth_input)

    return (eval_sess, resized_input_tensor, bottleneck_input, ground_truth_input, evaluation_step, prediction)


def save_graph_to_file(graph, graph_file_name, module_spec, class_count):
    """Saves a graph to file, creating a valid quantized one if necessary"""
    sess, _, _, _, _, _ = build_eval_session(module_spec, class_count)
    graph = sess.graph

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [ARGS.final_tensor_name]
    )

    with tf.gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())


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
            train_session, image_lists, ARGS.test_batch_size,
            'testing', ARGS.bottleneck_dir, ARGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor, ARGS.tfhub_module
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

    if ARGS.print_misclassified_test_images:
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


def main_func(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not ARGS.image_dir:
        tf.logging.error('Must set flag --image_dir.')
        return -1

    # Prepare necessary directories that can be used during training
    prepare_tensorboard_dirs()

    # Look at the img dir structure and create lists of all images
    image_lists = create_image_lists(ARGS.image_dir, ARGS.testing_percentage,
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
    do_distort_images = should_distort_images(
        ARGS.flip_left_right, ARGS.random_crop, ARGS.random_scale, ARGS.random_brightness
    )

    # Set up the pre-trained graph
    module_spec = hub.load_module_spec(ARGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (create_module_graph(module_spec))

    # Add the new layer that we'll be training
    with graph.as_default():
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_retrain_ops(
            class_count, ARGS.final_tensor_name, bottleneck_tensor, wants_quantization, is_training=True
        )

    with tf.Session(graph=graph) as sess:
        # Initialize all weights for the module to their pretrained values,
        # and for the newly added retraining layer to random initial values
        init = tf.global_variables_initializer()
        sess.run(init)

        # Set up the image decoding sub-graph
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

        if do_distort_images:
            # Image distortion has been enabled so set up distortion operations
            (distorted_jpeg_data_tensor, distorted_image_tensor) = add_input_distortions(
                ARGS.flip_left_right, ARGS.random_crop, ARGS.random_scale, ARGS.random_brightness, module_spec
            )
        else:
            # Calculate bottleneck image summaries and cache them on disk
            cache_bottlenecks(sess, image_lists, ARGS.image_dir,
                              ARGS.bottleneck_dir, jpeg_data_tensor,
                              decoded_image_tensor, resized_image_tensor,
                              bottleneck_tensor, ARGS.tfhub_module)

        # Create the operations we need to evaluate the accuracy of our new layer
        evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

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
                (train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(
                    sess, image_lists, ARGS.train_batch_size, 'training',
                    ARGS.image_dir, distorted_jpeg_data_tensor, distorted_image_tensor,
                    resized_image_tensor, bottleneck_tensor
                )
            else:
                (train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
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
                # practice this makes a negligable difference.
                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
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

                if (intermediate_frequency > 0 and (i % intermediate_frequency == 0) and i > 0):
                    # If we want to do an intermediate save, save a checkpoint of the train graph to restore into the eval graph
                    train_saver.save(sess, CHECKPOINT_NAME)
                    intermediate_file_name = (ARGS.intermediate_output_graphs_dir + 'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to: ' + intermediate_file_name)
                    save_graph_to_file(graph, intermediate_file_name, module_spec, class_count)

        # After training is complete, force one last save of the train checkpoint
        train_saver.save(sess, CHECKPOINT_NAME)

        # Training complete, so run a final test evaluation on some new images we haven't used before
        run_final_eval(sess, module_spec, class_count, image_lists, jpeg_data_tensor, decoded_image_tensor,
                       resized_image_tensor, bottleneck_tensor)

        # Write out the trained graph and labels with the weights stored as constants
        tf.logging.info('Save final result to : ' + ARGS.output_graph)
        if wants_quantization:
            tf.logging.info('The model is instrumented for quantization with TF-Lite')
        save_graph_to_file(graph, ARGS.output_graph, module_spec, class_count)
        with tf.gfile.FastGFile(ARGS.output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

        if ARGS.saved_model_dir:
            export_model(module_spec, class_count, ARGS.saved_model_dir)


if __name__=='__main__':

    # args - the input params needed for this script
    # unparsed_args - extra args or those needed for the tensorflow app call
    ARGS, unparsed_args = build_arg_parser().parse_known_args()
    tf.app.run(main=main_func, argv=[sys.argv[0]] + unparsed_args)
