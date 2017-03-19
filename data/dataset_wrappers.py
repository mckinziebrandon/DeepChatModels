import logging
import os

import tensorflow as tf
import pandas as pd
from data._dataset import Dataset
from utils import io_utils


class Cornell(Dataset):
    """Movie dialogs."""

    def __init__(self, data_dir, vocab_size=40000, max_seq_len=80):
        self._name = "cornell"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('CornellLogger')
        super(Cornell, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class Ubuntu(Dataset):
    """Technical support chat logs from IRC."""

    def __init__(self, data_dir, vocab_size=40000, max_seq_len=80):
        self._name = "ubuntu"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('UbuntuLogger')
        super(Ubuntu, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class WMT(Dataset):
    """English-to-French translation."""

    def __init__(self, data_dir, vocab_size=40000, max_seq_len=80):
        self._name = "wmt"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('WMTLogger')
        super(WMT, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class Reddit(Dataset):
    """Reddit comments from 2007-2015."""

    def __init__(self, data_dir, vocab_size=40000, max_seq_len=80):
        self._name = "reddit"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('RedditLogger')
        super(Reddit, self).__init__(data_dir, vocab_size=vocab_size, max_seq_len=max_seq_len)


class TestData(Dataset):
    """Mock dataset with a handful of sentences."""

    def __init__(self, data_dir, vocab_size=83, max_seq_len=80):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestDataLogger')
        self._name = "test_data"
        super(TestData, self).__init__(data_dir='/home/brandon/terabyte/Datasets/test_data',
                                       vocab_size=vocab_size,
                                       max_seq_len=max_seq_len)

