import logging
import tensorflow as tf
import os
from data._dataset import Dataset
from utils import io_utils


class TestData(Dataset):
    """Mock dataset with a handful of sentences."""

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestDataLogger')
        self._name = "test_data"
        self._data_dir = '/home/brandon/terabyte/Datasets/test_data'


        paths_triplet = io_utils.prepare_data(self._data_dir,
                                              self._data_dir + "/train_from.txt",
                                              self._data_dir + "/train_to.txt",
                                              self._data_dir + "/valid_from.txt",
                                              self._data_dir + "/valid_to.txt",
                                              100, 100)
        train_path, valid_path, vocab_path = paths_triplet
        self.paths = {}
        self.paths['from_train']    = train_path[0]
        self.paths['to_train']      = train_path[1]
        self.paths['from_valid']    = valid_path[0]
        self.paths['to_valid']      = valid_path[1]
        self.paths['from_vocab']    = vocab_path[0]
        self.paths['to_vocab']      = vocab_path[1]

        from_vocab_size = len(open(os.path.join(self._data_dir, "vocab100.from")).readlines())
        to_vocab_size = len(open(os.path.join(self._data_dir, "vocab100.to")).readlines())
        self.vocab_size = max(from_vocab_size, to_vocab_size)

        self._word_to_idx, _ = io_utils.get_vocab_dicts(self.paths['from_vocab'])
        _, self._idx_to_word = io_utils.get_vocab_dicts(self.paths['to_vocab'])
        assert len(self._word_to_idx) <= self.vocab_size
        assert len(self._idx_to_word) <= self.vocab_size

        self._train_data = self.read_data("train")
        self._valid_data = self.read_data("valid")
        self._train_size = len(self._train_data[0])
        self._valid_size = len(self._valid_data[0])

        # TODO: implemented this ridiculous fix for max_seq_len for time reasons,
        # BUT IT MUST GO IT IS GROSS
        enc, dec = self._train_data
        max_enc = max([len(s) for s in enc])
        max_dec = max([len(s) for s in dec])
        self._max_seq_len = max(max_enc, max_dec)


    @property
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        return self._word_to_idx

    @property
    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        return self._idx_to_word

    def as_words(self, token_ids):
        try:
            return " ".join([tf.compat.as_str(self._idx_to_word[i]) for i in token_ids])
        except IndexError:
            print("IndexError encountered for following sentence:\n", token_ids)
            print("\nVocab size is :", self.vocab_size)
            print("idx to word is:\n%r" % self._idx_to_word)
            print("With length %d" % len(self._idx_to_word))

    @property
    def data_dir(self):
        """Return path to directory that contains the data."""
        return self._data_dir

    @property
    def name(self):
        """Returns name of the dataset as a string."""
        return self._name

    def read_data(self, suffix="train"):
        source_data = []
        target_data = []
        # Counter for the number of source/target pairs that couldn't fit in _buckets.
        with tf.gfile.GFile(self.paths['from_%s' % suffix], mode="r") as source_file:
            with tf.gfile.GFile(self.paths['to_%s' % suffix], mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get source/target as list of word IDs.
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(io_utils.EOS_ID)
                    # Add to data_set and retrieve next id list.
                    source_data.append(source_ids)
                    target_data.append(target_ids)
                    source, target = source_file.readline(), target_file.readline()
        return (source_data, target_data)

    @property
    def train_size(self):
        return self._train_size

    @property
    def valid_size(self):
        return self._valid_size


    @property
    def train_data(self):
        """NEW FORMAT: returns a 2-tuple: (source_data, target_data) """
        return self._train_data

    @property
    def valid_data(self):
        """NEW FORMAT: returns a 2-tuple: (source_data, target_data) """
        return self._valid_data

    @property
    def max_seq_len(self):
        return self._max_seq_len
