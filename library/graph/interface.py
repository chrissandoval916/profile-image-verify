import abc


class GraphInterface(metaclass=abc.ABCMeta):

    GRAPH_NAME = ''

    @classmethod
    def set_graph_name(cls, file_name):
        cls.GRAPH_NAME = file_name

    @classmethod
    @abc.abstractmethod
    def create(cls, *args):
        """ Generate a graph object """

    @classmethod
    @abc.abstractmethod
    def save(cls):
        """ Save a graph to the set FILE_NAME file """
