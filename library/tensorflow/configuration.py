import tensorflow as tf
import collections
import hashlib
import os.path
import re

from library.utils import ensure_dir_exists

IMAGE_EXT = []
MAX_NUM_IMAGES_PER_CLASS = 0
SUMMARIES_DIR = ''
GRAPH_DIR = ''
LEARNING_RATE = 0
STORE_FREQUENCY = 0


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


def prepare_tensorboard_dirs():
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(SUMMARIES_DIR):
        tf.gfile.DeleteRecursively(SUMMARIES_DIR)
    tf.gfile.MakeDirs(SUMMARIES_DIR)
    if STORE_FREQUENCY > 0:
        ensure_dir_exists(GRAPH_DIR)
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
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in IMAGE_EXT:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(tf.gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning("No files found in {}. Make sure extension is one of the following: {}".format(dir_name, IMAGE_EXT))
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


def get_final_retrain_ops(class_count, final_tensor_name, bottleneck_tensor,
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
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)

    return train_step, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor