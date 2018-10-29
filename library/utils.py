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


