import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
import numpy as np
import time
from time import sleep
from utils.io_utils import *
from utils.config import TrainConfig
from chatbot import DynamicBot
from data import Cornell, Ubuntu, TestData

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Testing on WMT since it is the currentl largest dataset (22M lines).
# This is useful because it will be obvious, for memory reasons, how much the
# batch generating is really working.
WMT_DIR = '/home/brandon/terabyte/Datasets/wmt'
TRAIN_FILE = WMT_DIR + '/giga-fren.release2.fixed.en.ids40000'
TRAIN_SIZE = 22520376

def simple_batch_generator(batch_size):
    """Generates padded encoder,decoder batches. Work in progress.
    """

    def padded_batch(sentences, max_length):
        padded = np.array([s + [PAD_ID] * (max_length - len(s))
                           for s in sentences])
        return padded

    token_list = []
    with tf.gfile.GFile(TRAIN_FILE, mode="r") as source_file:
        source = source_file.readline()
        while source:
            token_list.append([int(x) for x in source.split()])
            if len(token_list) == batch_size:
                max_sent_len = max([len(s) for s in token_list])
                batch = padded_batch(token_list, max_sent_len)
                yield batch
                token_list = []
            source = source_file.readline()
        # Don't forget to yield the 'leftovers'!
        assert len(token_list) <= batch_size
        if len(token_list) > 0:
            max_sent_len = max([len(s) for s in token_list])
            batch = padded_batch(token_list, max_sent_len)
            yield batch


class TestGenerators(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestGeneratorsLogger')

    def test_fixed_num_batches(self):
        num_batches = 5
        self.batch_size = 8
        gen = simple_batch_generator(self.batch_size)
        for i in range(num_batches):
            print("Generating batch number", i)
            _ = next(gen)
            print("Waiting...")
            sleep(5)

    def test_full_wmt_simple(self):
        self.batch_size = 256
        gen = simple_batch_generator(self.batch_size)
        expected_num_batches = TRAIN_SIZE // self.batch_size
        i = 0
        for batch in gen:
            print("\rGenerated batch number %d/%d" % (i, expected_num_batches), end="")
            sys.stdout.flush()
            i += 1

if __name__ == '__main__':
    unittest.main()

