import tensorflow as tf

from data._dataset import Dataset
from utils import data_utils


class Ubuntu(Dataset):
    """Access to reformatted Ubuntu Dialogue Corpus for conversation training.
        TODO: Move reformatting functions from 'reformat_ubuntu' notebook to here.
    """

    def __init__(self, vocab_size):
        self._name = "ubuntu"
        self.vocab_size = vocab_size
        self._data_dir = '/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus'

        paths_triplet = data_utils.prepare_data(self._data_dir,
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

        encoder_dictionaries = data_utils.get_vocab_dicts(self.paths['from_vocab'])
        decoder_dictionaries = data_utils.get_vocab_dicts(self.paths['to_vocab'])

        self._word_to_idx, _ = encoder_dictionaries
        _, self._idx_to_word = decoder_dictionaries

        self._train_data = self.read_data("train")
        self._valid_data = self.read_data("valid")
        self._train_size = len(self._train_data)
        self._valid_size = len(self._valid_data)

    # ===================================================================
    # Required 'Dataset' method implementations:
    # ===================================================================

    def word_to_idx(self):
        """Encoder words to indices. """
        return self._word_to_idx

    def idx_to_word(self):
        """Decoder words to indices. """
        return self._idx_to_word

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def name(self):
        return self._name

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
        return 100

    # ===================================================================
    # Additional methods:
    # ===================================================================

    def read_data(self, suffix="train"):
        # TODO: Move to data_utils. Duplicated in other datasets.
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
                    target_ids.append(data_utils.EOS_ID)
                    # Add to data_set and retrieve next id list.
                    source_data.append(source_ids)
                    target_data.append(target_ids)
                    source, target = source_file.readline(), target_file.readline()
        return (source_data, target_data)
