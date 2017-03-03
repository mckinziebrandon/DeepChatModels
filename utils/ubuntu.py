from utils.dataset import Dataset
import utils.data_utils
from tensorflow import gfile

class Ubuntu(Dataset):

    def __init__(self, vocab_size):
        self._name = "ubuntu"
        self.vocab_size = vocab_size
        self._data_dir = '/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus'

        paths_triplet = utils.data_utils.prepare_data(self._data_dir,
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


    # ===================================================================
    # Required 'Dataset' method implementations:
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

    # ===================================================================
    # Additional methods:
    # ===================================================================

