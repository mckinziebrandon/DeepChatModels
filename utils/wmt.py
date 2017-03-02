from utils.dataset import Dataset
from utils.data_utils import *
from tensorflow import gfile

class WMT(Dataset):

    def __init__(self, from_vocab_size=40000, to_vocab_size=None):
        self.name = "wmt"

        if to_vocab_size == None:
            to_vocab_size = from_vocab_size
        self._data_dir = '/home/brandon/terabyte/Datasets/wmt'

        # Get wmt data to the specified directory.
        train_path = os.path.join(self._data_dir, "giga-fren.release2.fixed")
        dev_path = os.path.join(self._data_dir, "newstest2013")

        paths_triplet = prepare_data(self._data_dir,
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


