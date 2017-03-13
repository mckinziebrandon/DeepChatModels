import logging

import tensorflow as tf
import pandas as pd
from data._dataset import Dataset
from utils import io_utils


class Cornell(Dataset):

    def __init__(self, data_dir, vocab_size=20000, max_seq_len=80):
        self._name = "cornell"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('CornellLogger')
        #self._data_dir = '/home/brandon/terabyte/Datasets/cornell_movie_corpus'
        super(Cornell, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class Ubuntu(Dataset):

    def __init__(self, data_dir, vocab_size=20000, max_seq_len=80):
        self._name = "ubuntu"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('UbuntuLogger')
        super(Ubuntu, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class WMT(Dataset):

    def __init__(self, data_dir, vocab_size=40000, max_seq_len=80):
        self._name = "wmt"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('WMTLogger')
        super(WMT, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)

