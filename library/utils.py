import os.path
import tensorflow as tf

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