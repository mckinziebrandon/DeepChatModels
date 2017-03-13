import logging

import tensorflow as tf
import pandas as pd
from data._dataset import Dataset
from utils import io_utils


class Cornell(Dataset):

    def __init__(self, data_dir, vocab_size=20000):

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('CornellLogger')

        self._name = "cornell"
        self.vocab_size = vocab_size
        #self._data_dir = '/home/brandon/terabyte/Datasets/cornell_movie_corpus'
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
        return self._generator(self.paths['from_train'], self.paths['to_train'], batch_size)

    def valid_generator(self, batch_size):
        return self._generator(self.paths['from_valid'], self.paths['to_valid'], batch_size)

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
    def train_data(self):
        """Removed. Use generator instead."""
        self.log.error("Tried getting full training data. Use train_generator instead.")
        raise NotImplemented

    @property
    def valid_data(self):
        """Removed. Use generator instead."""
        self.log.error("Tried getting full validation data. Use valid_generator instead.")
        raise NotImplemented

    @property
    def max_seq_len(self):
        self.log.error("Tried getting max_seq_len. Unsupported since DynamicBot.")
        raise NotImplemented

    # =========================================================================================
    # Deprecated methods that have been replaced by better/more efficent ones.
    # Keeping in case users want to load full dataset at once/don't care about memory usage.
    # =========================================================================================

    def read_data(self, suffix="train"):
        return self._read_data(self.paths['from_%s' % suffix], self.paths['to_%s' % suffix])

