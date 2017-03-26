import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
import numpy as np
import time
import yaml
from utils.io_utils import *
from chatbot import DynamicBot
from data import Cornell, Ubuntu, TestData

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
BASE='/home/brandon/Documents/seq2seq_projects/tests'


class TestConfig(unittest.TestCase):
    """Test behavior of tf.contrib.rnn after migrating to r1.0."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestRNNLogger')

    def test_path(self):
        conf_path = os.path.abspath('../basic_config.yml')
        with tf.gfile.GFile(conf_path) as file:
            config_dict = yaml.load(file)

        self.assertIsInstance(config_dict, dict, "Config is type %r" % type(config_dict))

        for key in config_dict['model_params']:
            self.log.info(key)
            self.assertIsNotNone(key)

if __name__ == '__main__':
    unittest.main()

