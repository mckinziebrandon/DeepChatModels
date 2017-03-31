import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
import numpy as np
import time
import yaml
from utils import io_utils
from chatbot import DynamicBot
from data import Cornell, Ubuntu, TestData

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

# Allow user to override config values with command-line args.
# All test_flags with default as None are not accessed unless set.
test_flags = tf.app.flags
test_flags.DEFINE_string("config", "configs/default.yml", "path to config (.yml) file. Defaults to DynamicBot on Cornell.")
test_flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
test_flags.DEFINE_string("model_params", "{}", "")
test_flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
test_flags.DEFINE_string("dataset_params", "{}", "")
TEST_FLAGS = test_flags.FLAGS
HERE = os.path.dirname(os.path.realpath(__file__))

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

    def test_merge_params(self):
        """Checks how parameters passed to FLAGS interact with
        parameters from yaml files. Expected behavior is that any
        params in FLAGS will override those from files, but that
        all values from file will be used if not explicitly passed to
        FLAGS.
        """

        # Set values to typical use case.
        TEST_FLAGS.config = os.path.join(HERE, '../', 'configs/default.yml')
        TEST_FLAGS.model = "chatbot.ChatBot"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['model'], 'chatbot.ChatBot')

        TEST_FLAGS.model = "chatbot.DynamicBot"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['model'], 'chatbot.DynamicBot')

        TEST_FLAGS.dataset = "data.TestData"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['dataset'], 'data.TestData')


if __name__ == '__main__':
    unittest.main()

