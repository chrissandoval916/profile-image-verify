import tensorflow as tf
import random
import numpy as np
import os.path

from library.utils import get_image_path, ensure_dir_exists

MAX_NUM_IMAGES_PER_CLASS = 0


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