""" ABC for datasets. """
from abc import ABCMeta, abstractmethod, abstractproperty

class Dataset(metaclass=ABCMeta):

    @abstractmethod
    def train_generator(self, batch_size):
        """Returns a generator function for batches of batch_size train data."""
        pass

    @abstractmethod
    def valid_generator(self, batch_size):
        """Returns a generator function for batches of batch_size validation data."""
        pass

    @abstractproperty
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        pass

    @abstractproperty
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

    @abstractproperty
    def train_size(self):
        """Returns number of samples (sentences) in this dataset."""
        pass

    @abstractproperty
    def valid_size(self):
        """Returns number of samples (sentences) in this dataset."""
        pass

    @abstractproperty
    def train_data(self):
        """List of training samples (token IDs)."""
        pass

    @abstractproperty
    def valid_data(self):
        """List of validation samples (token IDs)."""
        pass

    @abstractproperty
    def max_seq_len(self):
        """Return the number of tokens in the longest example"""
        pass


