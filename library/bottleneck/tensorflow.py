import tensorflow as tf
import random
import numpy as np
import os.path

from library.bottleneck.interface import BottleneckInterface
from library.utils import ensure_dir_exists


class TensorflowBottleneck(BottleneckInterface):

    MAX_NUM_IMAGES_PER_CLASS = 0
    TFHUB_MODULE = ''
    BOTTLENECK_FILE_DIR = ''
    IMAGE_FILE_DIR = ''

    def __init__(self, max_num_images_per_class, tfhub_module, bottleneck_file_dir, image_file_dir):
        self.MAX_NUM_IMAGES_PER_CLASS = max_num_images_per_class
        self.TFHUB_MODULE = tfhub_module
        self.BOTTLENECK_FILE_DIR = bottleneck_file_dir
        self.IMAGE_FILE_DIR = image_file_dir

    def create(self, bottleneck_path, image_lists, label_name, index,
               category, session, image_data_tensor,
               decoded_image_tensor, resized_input_tensor,
               bottleneck_tensor):
        """
        Create a single bottleneck file
        :param bottleneck_path: File system path string to bottleneck file
        :param image_lists: OrderedDict of training images for each label
        :param label_name: Label string we want to get an image for
        :param index: Integer offset of the image we want. This will be modulo-ed by the
                      available number of images for the label, so it can be arbitrarily large
        :param category: Name string of which set to pull images from (training, testing, validation)
        :param session: Current active TensorFlow Session
        :param image_data_tensor: Input tensor for jpeg data from file
        :param decoded_image_tensor: Output of decoding and resizing the image
        :param resized_input_tensor: Input node of the recognition graph
        :param bottleneck_tensor: Penultimate output layer of the graph
        :return:
        """
        tf.logging.info('Creating bottleneck at ' + bottleneck_path)
        image_path = self.get_image_path(image_lists, label_name, index, category)

        if not tf.gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        try:
            bottleneck_values = self.run_bottleneck_on_image(session, image_data, image_data_tensor,
                                                             decoded_image_tensor, resized_input_tensor,
                                                             bottleneck_tensor)
        except Exception as e:
            raise RuntimeError('Error during processing file %s (%s)' % (image_path, str(e)))
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    def save(self, session, image_lists, image_tensor_placeholder, manipulated_image_data,
                      resized_input_tensor, bottleneck_tensor):
        """
        If there are no distortions applied, we are very likely to read the same image
        multiple times. As such we want to only calculate the bottleneck layer values once
        for each image during preprocessing and then just read the cached values repeatedly
        during training. This function goes through all the images, calculates the values,
        and then caches them
        :param session: Current active TensorFlow Session
        :param image_lists: OrderedDict of training images for each label
        :param image_tensor_placeholder: Input tensor for jpeg data from file
        :param manipulated_image_data: Output of decoding and resizing the image
        :param resized_input_tensor: Input node of the recognition graph
        :param bottleneck_tensor: Penultimate output layer of the graph
        :return:
        """
        how_many_bottlenecks = 0
        ensure_dir_exists(self.BOTTLENECK_FILE_DIR)
        for label_name, label_lists in image_lists.items():
            for category in ['training', 'testing', 'validation']:
                category_list = label_lists[category]
                for index, unused_base_name in enumerate(category_list):
                    self.get_bottleneck_values(session, image_lists, label_name, index,
                                               category, image_tensor_placeholder,
                                               manipulated_image_data, resized_input_tensor,
                                               bottleneck_tensor)

                    how_many_bottlenecks += 1
                    if how_many_bottlenecks % 100 == 0:
                        tf.logging.info(str(how_many_bottlenecks) + ' bottleneck files created.')

    def get(self, session, image_lists, batch_size, category, image_tensor_placeholder, manipulated_image_data,
            resized_input_tensor, bottleneck_tensor, distorting_images=False):
        if distorting_images:
            return self.get_distorted_bottlenecks(session, image_lists, batch_size, category, image_tensor_placeholder,
                                                  manipulated_image_data, resized_input_tensor, bottleneck_tensor)
        else:
            return self.get_cached_bottlenecks(session, image_lists, batch_size, category, image_tensor_placeholder,
                                               manipulated_image_data, resized_input_tensor, bottleneck_tensor)

    @staticmethod
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

    @staticmethod
    def get_relative_path(image_lists, label_name, index, category):
        """
        Return a relative path to a file for a label at the given index
        :param image_lists: OrderedDict of training images for each label
        :param label_name: Label string we want to get an image for
        :param index: Integer offset of the image we want. This will be modulo-ed by the
                      available number of images for the label, so it can be arbitrarily large
        :param category: Name string of which set to pull images from (training, testing, validation)
        :return: Relative file path string for a file that meets the requested parameters
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
        return os.path.join(sub_dir, base_name)

    def parse_module_name(self):
        return (self.TFHUB_MODULE.replace('://', '~')  # URL scheme.
                .replace('/', '~')  # URL and Unix paths.
                .replace(':', '~').replace('\\', '~'))  # Windows paths.

    def get_bottleneck_path(self, image_lists, label_name, index, category):
        """
        Return a path to a bottleneck file for a label at the given index
        :param image_lists: OrderedDict of training images for each label
        :param label_name: Label string we want to get an image for
        :param index: Integer offset of the image we want. This will be modulo-ed by the
                      available number of images for the label, so it can be arbitrarily large
        :param category: Name string of which set to pull images from (training, testing, validation)
        :return: File system path string to an image that meets the requested parameters
        """
        module_name = self.parse_module_name()
        relative_path = self.get_relative_path(image_lists, label_name, index, category)
        full_path = os.path.join(self.BOTTLENECK_FILE_DIR, relative_path)

        return full_path + '_' + module_name + '.txt'

    def get_image_path(self, image_lists, label_name, index, category):
        """
        Return a path to an image for a label at the given index
        :param image_lists: OrderedDict of training images for each label
        :param label_name: Label string we want to get an image for
        :param index: Integer offset of the image we want. This will be modulo-ed by the
                      available number of images for the label, so it can be arbitrarily large
        :param category: Name string of which set to pull images from (training, testing, validation)
        :return:
        """
        relative_path = self.get_relative_path(image_lists, label_name, index, category)
        full_path = os.path.join(self.IMAGE_FILE_DIR, relative_path)
        return full_path

    def get_distorted_bottlenecks(self, session, image_lists, batch_size, category,
                                  image_tensor_placeholder, manipulated_image_data, resized_input_tensor,
                                  bottleneck_tensor):
        """
        If training with distortion (crops,scales,flips,etc) we have to recalculate the full model
        for every image, and so we cant used cached bottleneck values. Instead we find random images
        for the requested category, run them through the distortion graph, and then the full graph
        to get the bottleneck results for each
        :param session: Current TensorFlow Session
        :param image_lists: OrderedDict of training images for each label
        :param batch_size: Integer number for the number of bottleneck values to return
        :param category: Name string of which set to pull images from (training, testing, validation)
        :param image_tensor_placeholder: Input layer we feed the image data to
        :param manipulated_image_data: Output node of the distortion graph
        :param resized_input_tensor: Input node of the recognition graph
        :param bottleneck_tensor: Bottleneck output layer of the CNN graph
        :return: List of bottleneck arrays and their corresponding ground truths
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        for unused_i in range(batch_size):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
            image_path = self.get_image_path(image_lists, label_name, image_index, category)

            if not tf.gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image_path)
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()
            # TODO possible optimization point
            # Note that we materialize the distorted_image_data as a numpy array
            # before sending running inference on the image. This involves 2 memory
            # copies and might be optimized in other implements
            distorted_image_data = session.run(manipulated_image_data, {image_tensor_placeholder: image_data})
            bottleneck_values = session.run(bottleneck_tensor, {resized_input_tensor: distorted_image_data})
            bottleneck_values = np.squeeze(bottleneck_values)
            bottlenecks.append(bottleneck_values)
            ground_truths.append(label_index)
        return bottlenecks, ground_truths

    def get_cached_bottlenecks(self, session, image_lists, batch_size, category,
                               image_tensor_placeholder, manipulated_image_data, resized_input_tensor,
                               bottleneck_tensor):
        """
        If no distortions are being applied, use the cached bottleneck values directly from disk. Picks
        a random set of images from the specified category
        :param session: Current TensorFlow Session
        :param image_lists: OrderedDict of training images for each label
        :param batch_size: Integer number for the number of bottleneck values to return
        :param category: Name string of which set to pull images from (training, testing, validation)
        :param image_tensor_placeholder: Layer to feed teh image data into
        :param manipulated_image_data: Output of decoding and resizing the image
        :param resized_input_tensor: Input node of the recognition graph
        :param bottleneck_tensor: Bottleneck output layer of the CNN graph
        :return: List of bottleneck arrays, their corresponding ground truths, and the
        relevant file names.
        """
        class_count = len(image_lists.keys())
        bottlenecks = []
        ground_truths = []
        filenames = []
        if batch_size >= 0:
            # Retrieve a random sample of bottlenecks
            for unused_i in range(batch_size):
                label_index = random.randrange(class_count)
                label_name = list(image_lists.keys())[label_index]
                image_index = random.randrange(self.MAX_NUM_IMAGES_PER_CLASS + 1)
                image_name = self.get_image_path(image_lists, label_name, image_index, category)
                bottleneck = self.get_bottleneck_values(session, image_lists, label_name, image_index,
                                                        category, image_tensor_placeholder,
                                                        manipulated_image_data, resized_input_tensor,
                                                        bottleneck_tensor)
                bottlenecks.append(bottleneck)
                ground_truths.append(label_index)
                filenames.append(image_name)
        else:
            # Retrieve all bottlenecks
            for label_index, label_name in enumerate(image_lists.keys()):
                for image_index, image_name in enumerate(image_lists[label_name][category]):
                    image_name = self.get_image_path(image_lists, label_name, image_index, category)
                    bottleneck = self.get_bottleneck_values(session, image_lists, label_name, image_index,
                                                            category, image_tensor_placeholder,
                                                            manipulated_image_data, resized_input_tensor,
                                                            bottleneck_tensor)
                    bottlenecks.append(bottleneck)
                    ground_truths.append(label_index)
                    filenames.append(image_name)
        return bottlenecks, ground_truths, filenames

    def get_bottleneck_values(self, session, image_lists, label_name, index,
                              category, image_tensor_placeholder, manipulated_image_data,
                              resized_input_tensor, bottleneck_tensor):
        """
        If a cached version of the bottleneck data exists, return it. Else generate and save
        :param session: Current active TensorFlow Session
        :param image_lists: OrderedDict of training images for each label
        :param label_name: Label string we want to get an image for
        :param index: Integer offset of the image we want. This will be modulo-ed by the
                      available number of images for the label, so it can be arbitrarily large
        :param category: Name string of which set to pull images from (training, testing, validation)
        :param image_tensor_placeholder: Input tensor for jpeg data from file
        :param manipulated_image_data: Output of decoding and resizing the image
        :param resized_input_tensor: Input node of the recognition graph
        :param bottleneck_tensor: Penultimate output layer of the graph
        :return: Numpy array of values produced by the bottleneck layer for the image
        """
        label_lists = image_lists[label_name]
        sub_dir = label_lists['dir']
        sub_dir_path = os.path.join(self.BOTTLENECK_FILE_DIR, sub_dir)
        ensure_dir_exists(sub_dir_path)
        bottleneck_path = self.get_bottleneck_path(image_lists, label_name, index, category)
        if not os.path.exists(bottleneck_path):
            self.create(bottleneck_path, image_lists, label_name, index, category, session,
                        image_tensor_placeholder, manipulated_image_data, resized_input_tensor,
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
            self.create(bottleneck_path, image_lists, label_name, index, category, session,
                        image_tensor_placeholder, manipulated_image_data, resized_input_tensor,
                        bottleneck_tensor)
            with open(bottleneck_path, 'r') as bottleneck_file:
                bottleneck_string = bottleneck_file.read()
            # Dont worry about exceptions as they shouldn't happen after a fresh creation
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        return bottleneck_values
