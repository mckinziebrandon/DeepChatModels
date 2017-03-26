"""Named data wrapper classes. No added functionality to dataset base class for now,
but preprocessing checks will be incorporated into each when it's time.
"""

import os
import logging
import tensorflow as tf
import pandas as pd
from data._dataset import Dataset
from utils import io_utils


class Cornell(Dataset):
    """Movie dialogs."""

    def __init__(self, dataset_params):
        self._name = "cornell"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('CornellLogger')
        super(Cornell, self).__init__(dataset_params)


class Ubuntu(Dataset):
    """Technical support chat logs from IRC."""

    def __init__(self, dataset_params):
        self._name = "ubuntu"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('UbuntuLogger')
        super(Ubuntu, self).__init__(dataset_params)


class WMT(Dataset):
    """English-to-French translation."""

    def __init__(self, dataset_params):
        self._name = "wmt"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('WMTLogger')
        super(WMT, self).__init__(dataset_params)


class Reddit(Dataset):
    """Reddit comments from 2007-2015."""

    def __init__(self, dataset_params):
        self._name = "reddit"
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('RedditLogger')
        super(Reddit, self).__init__(dataset_params)


class TestData(Dataset):
    """Mock dataset with a handful of sentences."""

    def __init__(self, dataset_params):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestDataLogger')
        self._name = "test_data"
        super(TestData, self).__init__(dataset_params)
