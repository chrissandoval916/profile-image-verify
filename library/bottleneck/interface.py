import abc


class BottleneckInterface(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def create(cls, *args):
        """ Create a new bottleneck"""

    @classmethod
    @abc.abstractmethod
    def save(cls, *args):
        """ Save a bottleneck"""

    @classmethod
    @abc.abstractmethod
    def get(cls, *args):
        """ Get a bottleneck """
