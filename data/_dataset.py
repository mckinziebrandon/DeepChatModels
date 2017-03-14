""" ABC for datasets. """
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import io_utils
import random
from utils.io_utils import EOS_ID, PAD_ID
import logging


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
    def max_seq_len(self):
        """Return the number of tokens in the longest example"""
        pass


class Dataset(DatasetABC):

    def __init__(self, data_dir, vocab_size=20000, max_seq_len=80):
        """Implements the most general of subset of operations that all classes can use."""
        self._max_seq_len = max_seq_len
        print("max_seq_len recorded as ", max_seq_len)
        self.vocab_size = vocab_size
        self._data_dir = data_dir
        # We query io_utils to ensure all data files are organized properly,
        # and io_utils returns the paths to files of interest.
        paths_triplet = io_utils.prepare_data(self._data_dir,
                                              self._data_dir + "/train_from.txt",
                                              self._data_dir + "/train_to.txt",
                                              self._data_dir + "/valid_from.txt",
                                              self._data_dir + "/valid_to.txt",
                                              vocab_size, vocab_size)

        train_path, valid_path, vocab_path = paths_triplet
        self.paths = {}
        self.paths['from_train']    = train_path[0]
        self.paths['to_train']      = train_path[1]
        self.paths['from_valid']    = valid_path[0]
        self.paths['to_valid']      = valid_path[1]
        self.paths['from_vocab']    = vocab_path[0]
        self.paths['to_vocab']      = vocab_path[1]

        self._word_to_idx, _ = io_utils.get_vocab_dicts(self.paths['from_vocab'])
        _, self._idx_to_word = io_utils.get_vocab_dicts(self.paths['to_vocab'])

    def train_generator(self, batch_size):
        """Returns a generator function. Each call to next() yields a batch
            of size batch_size data as numpy array of shape [batch_size, max_seq_len],
            where max_seq_len is the longest sentence in the returned batch.
        """
        return self._generator(self.paths['from_train'], self.paths['to_train'],
                               batch_size)

    def valid_generator(self, batch_size):
        return self._generator(self.paths['from_valid'], self.paths['to_valid'], batch_size)

    def _generator(self, from_path, to_path, batch_size):
        """Returns a generator function that reads data from file, an d
            yields shuffled batches.

        Args:
            from_path: full path to file for encoder inputs.
            to_path: full path to file for decoder inputs.
            batch_size: number of samples to yield at once.
        """

        def longest_sentence(enc_list, dec_list):
            max_enc_len = max([len(s) for s in enc_list])
            max_dec_len = max([len(s) for s in dec_list])
            return max(max_enc_len, max_dec_len)

        def padded_batch(encoder_tokens, decoder_tokens):
            max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
            encoder_batch = np.array([s + [PAD_ID] * (max_sent_len - len(s)) for s in encoder_tokens])[:, ::-1]
            decoder_batch = np.array([s + [PAD_ID] * (max_sent_len - len(s)) for s in decoder_tokens])
            return encoder_batch, decoder_batch

        encoder_tokens = []
        decoder_tokens = []
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:

                source, target = source_file.readline(), target_file.readline()
                while source and target:

                    # Skip any sentence pairs that are too long for user specifications.
                    space_needed = max(len(source.split()), len(target.split()))
                    if space_needed > self.max_seq_len:
                        source, target = source_file.readline(), target_file.readline()
                        continue

                    # Reformat token strings to token lists.
                    # Note: GO_ID is prepended by the chat bot, since it determines
                    # whether or not it's responsible for responding.
                    encoder_tokens.append([int(x) for x in source.split()])
                    decoder_tokens.append([int(x) for x in target.split()] + [EOS_ID])

                    # Have we collected batch_size number of sentences? If so, pad & yield.
                    assert len(encoder_tokens) == len(decoder_tokens)
                    if len(encoder_tokens) == batch_size:
                        yield padded_batch(encoder_tokens, decoder_tokens)
                        encoder_tokens = []
                        decoder_tokens = []
                    source, target = source_file.readline(), target_file.readline()

                # Don't forget to yield the 'leftovers'!
                assert len(encoder_tokens) == len(decoder_tokens)
                assert len(encoder_tokens) <= batch_size
                if len(encoder_tokens) > 0:
                    yield padded_batch(encoder_tokens, decoder_tokens)

    @property
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        return self._word_to_idx

    @property
    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        return self._idx_to_word

    def as_words(self, sentence):
        """Initially, this function was just the one-liner below:

            return " ".join([tf.compat.as_str(self._idx_to_word[i]) for i in sentence])

            Since then, it has become apparent that some character aren't converted properly,
            and tf has issues decoding. In (rare) cases that this occurs, I've setup the
            try-catch block to help inspect the root causes. It will remain here until the
            problem has been adequately diagnosed.
        """
        words = []
        try:
            for idx, i in enumerate(sentence):
                w = self._idx_to_word[i]
                w_str = tf.compat.as_str(w)
                words.append(w_str)
            return " ".join(words)
            #return " ".join([tf.compat.as_str(self._idx_to_word[i]) for i in sentence])
        except UnicodeDecodeError  as e:
            print("Error: ", e)
            print("Final index:", idx, "and token:", i)
            print("Final word: ", self._idx_to_word[i])
            print("Sentence length:", len(sentence))
            print("\n\nIndexError encountered for following sentence:\n", sentence)
            print("\nVocab size is :", self.vocab_size)
            print("Words:", words)

    @property
    def data_dir(self):
        """Return path to directory that contains the data."""
        return self._data_dir

    @property
    def name(self):
        """Returns name of the dataset as a string."""
        return self._name

    @property
    def train_size(self):
        raise NotImplemented

    @property
    def valid_size(self):
        raise NotImplemented

    @property
    def max_seq_len(self):
        return self._max_seq_len

