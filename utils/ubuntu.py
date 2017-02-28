from utils import Dataset
from utils.data_utils import prepare_ubuntu_data
from tensorflow import gfile

class Ubuntu(Dataset):

    def __init__(self, vocab_size):

        self._data_dir = '/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus'
        train, dev, _ = prepare_ubuntu_data(self._data_dir, vocab_size)

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

