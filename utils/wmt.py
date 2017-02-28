from utils import Dataset
from utils.data_utils import *
from tensorflow import gfile

class WMT(Dataset):

    def __init__(self, from_vocab_size, to_vocab_size=None):

        if to_vocab_size == None:
            to_vocab_size = from_vocab_size

        self._data_dir = '/home/brandon/terabyte/Datasets/wmt'
        train, dev, _ = prepare_wmt_data(self._data_dir, from_vocab_size, to_vocab_size)

        self.from_train, self.to_train = train
        self.from_dev, self.to_dev     = dev

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

    # ===================================================================
    # Additional methods.
    # ===================================================================

    def open_train_file(self, from_or_to):
        if from_or_to == "from":
            return gfile.GFile(self.from_train, mode="r")
        else:
            return gfile.GFile(self.to_train, mode="r")

