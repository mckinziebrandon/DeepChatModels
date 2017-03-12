""" ABC for datasets. """
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty
from utils.io_utils import EOS_ID, PAD_ID

class DatasetABC(metaclass=ABCMeta):

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


class Dataset(DatasetABC):
    """Implements the most general of subset of operations that all classes can use."""

    def _generator(self, from_path, to_path, batch_size):

        def padded_batch(sentences, max_length):
            padded = np.array([s + [PAD_ID] * (max_length - len(s)) for s in sentences])
            return padded

        def longest_sentence(enc_list, dec_list):
            max_enc_len = max([len(s) for s in enc_list])
            max_dec_len = max([len(s) for s in dec_list])
            return max(max_enc_len, max_dec_len)

        encoder_tokens = []
        decoder_tokens = []
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get the next sentence as integer token IDs.
                    encoder_tokens.append([int(x) for x in source.split()])
                    decoder_tokens.append([int(x) for x in target.split()] + [EOS_ID])
                    # Have we collected batch_size number of sentences? If so, pad & yield.
                    assert len(encoder_tokens) == len(decoder_tokens)
                    if len(encoder_tokens) == batch_size:
                        max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
                        # Encoder sentences are fed in reverse intentionally.
                        encoder_batch = padded_batch(encoder_tokens, max_sent_len)[:, ::-1]
                        decoder_batch = padded_batch(decoder_tokens, max_sent_len)
                        yield encoder_batch, decoder_batch
                        # Clear token containers for next batch.
                        encoder_tokens = []
                        decoder_tokens = []
                    source, target = source_file.readline(), target_file.readline()
                # Don't forget to yield the 'leftovers'!
                assert len(encoder_tokens) == len(decoder_tokens)
                assert len(encoder_tokens) <= batch_size
                if len(encoder_tokens) > 0:
                    max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
                    encoder_batch = padded_batch(encoder_tokens, max_sent_len)[:, ::-1]
                    decoder_batch = padded_batch(decoder_tokens, max_sent_len)
                    yield encoder_batch, decoder_batch

    def _read_data(self, from_path, to_path):
        """(Deprecated, use generator methods instead).
        Read entire dataset into memory. Can be prohibitively memory intensive for large
        datasets.
        Args:
            suffix: (str) either "train" or "valid"

        Returns:
            2-tuple (source_data, target_data) of sentence-token lists.
        """
        source_data = []
        target_data = []
        # Counter for the number of source/target pairs that couldn't fit in _buckets.
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get source/target as list of word IDs.
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(EOS_ID)
                    # Add to data_set and retrieve next id list.
                    source_data.append(source_ids)
                    target_data.append(target_ids)
                    source, target = source_file.readline(), target_file.readline()
        return source_data, target_data

    def train_generator(self, batch_size):
        """Returns a generator function for batches of batch_size train data."""
        raise NotImplemented

    def valid_generator(self, batch_size):
        """Returns a generator function for batches of batch_size validation data."""
        raise NotImplemented

    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        raise NotImplemented

    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        raise NotImplemented

    def data_dir(self):
        """Return path to directory that contains the data."""
        raise NotImplemented

    def name(self):
        """Returns name of the dataset as a string."""
        raise NotImplemented

    def train_size(self):
        """Returns number of samples (sentences) in this dataset."""
        raise NotImplemented

    def valid_size(self):
        """Returns number of samples (sentences) in this dataset."""
        raise NotImplemented

    def train_data(self):
        """List of training samples (token IDs)."""
        raise NotImplemented

    def valid_data(self):
        """List of validation samples (token IDs)."""
        raise NotImplemented

    def max_seq_len(self):
        """Return the number of tokens in the longest example"""
        raise NotImplemented
