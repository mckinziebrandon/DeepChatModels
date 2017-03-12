from data._dataset import Dataset
from utils import io_utils
import logging
import tensorflow as tf
import os


class WMT(Dataset):

    def __init__(self, from_vocab_size, to_vocab_size=None):

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('WMTLogger')

        self._name = "wmt"
        self.vocab_size = from_vocab_size
        if to_vocab_size == None:
            to_vocab_size = from_vocab_size
        self._data_dir = '/home/brandon/terabyte/Datasets/wmt'

        # Get wmt data to the specified directory.
        train_path = os.path.join(self._data_dir, "giga-fren.release2.fixed")
        dev_path = os.path.join(self._data_dir, "newstest2013")

        paths_triplet = io_utils.prepare_data(self._data_dir,
                                     train_path + ".en",
                                     train_path + ".fr",
                                     dev_path + ".en",
                                     dev_path + ".fr",
                                     from_vocab_size, to_vocab_size)

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

    def word_to_idx(self):
        raise NotImplemented

    def idx_to_word(self):
        raise NotImplemented

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def name(self):
        return self._name

    def as_words(self, sentence):
        import sys
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

    def read_data(self, suffix="train"):
        return self._read_data(self.paths['from_%s' % suffix], self.paths['to_%s' % suffix])
