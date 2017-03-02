""" ABC for datasets. """
from abc import ABCMeta, abstractmethod, abstractproperty

# TODO: Require a method that returns a generator over data samples.
class Dataset(metaclass=ABCMeta):
    @abstractmethod
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        pass

    @abstractmethod
    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        pass

    @abstractproperty
    def data_dir(self):
        """Return path to directory that contains the data."""
        pass

    @abstractproperty
    def name(self):
        """Returns name of the dataset as a string."""
        pass

