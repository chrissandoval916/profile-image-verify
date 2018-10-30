import abc


class BottleneckInterface(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def create(cls, *args):
        """ Create a new bottleneck """

    @classmethod
    @abc.abstractmethod
    def run(cls, *args):
        """ Runs inference on an image to generate bottleneck layer """

    @classmethod
    @abc.abstractmethod
    def get(cls, *args):
        """ Get a bottleneck """

    @classmethod
    def get_batch(cls, *args):
        """ Get a batch of bottlenecks """
