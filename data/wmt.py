from data._dataset import Dataset
from utils import io_utils
import pandas as pd
import tensorflow as tf
import os


class WMT(Dataset):

    def __init__(self, from_vocab_size, to_vocab_size=None):

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

        self._train_data = self.read_data("train")
        self._valid_data = self.read_data("valid")
        self._train_size = len(self._train_data[0])
        self._valid_size = len(self._valid_data[0])

        # TODO: implemented this ridiculous fix for max_seq_len for time reasons,
        # BUT IT MUST GO IT IS GROSS
        enc, dec = self._train_data
        max_enc = max([len(s) for s in enc])
        max_dec = max([len(s) for s in dec])
        self._df_lengths = pd.DataFrame({'EncoderLengths': [len(s) for s in enc],
                                         'DecoderLengths': [len(s) for s in dec]})
        self._max_seq_len = max(max_enc, max_dec)

    # ===================================================================
    # Required 'Dataset' method implementations.
    # ===================================================================

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

    def describe(self):
        print("------------------ Description: Cornell Movie Dialogues -------------------")
        print("Training data has %d samples." % self._train_size)
        print("Validation data has %d samples." % self._valid_size)
        print("Longest sequence is %d words long." % self._max_seq_len)


