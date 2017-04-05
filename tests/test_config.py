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
from chatbot.globals import DEFAULT_FULL_CONFIG
dir = os.path.dirname(os.path.realpath(__file__))

# Allow user to override config values with command-line args.
# All test_flags with default as None are not accessed unless set.
test_flags = tf.app.flags
test_flags.DEFINE_string("config", "macros/test_config.yml", "path to config (.yml) file.")
test_flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
test_flags.DEFINE_string("model_params", "{}", "")
test_flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
test_flags.DEFINE_string("dataset_params", "{}", "")
TEST_FLAGS = test_flags.FLAGS
KEYS = ['model', 'model_params', 'dataset', 'dataset_params']


def reset_flags(flags):
    """Reset flags to the default values."""
    for name in KEYS:
        setattr(flags, name, '{}')


class TestConfig(unittest.TestCase):
    """Test behavior of tf.contrib.rnn after migrating to r1.0."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestRNNLogger')
        with open('macros/test_config.yml') as f:
            # So we can always reference default vals.
            self.test_config = yaml.load(f)

    def test_path(self):
        conf_path = os.path.abspath('macros/test_config.yml')
        with tf.gfile.GFile(conf_path) as file:
            config_dict = yaml.load(file)

        self.assertIsInstance(config_dict, dict, "Config is type %r" % type(config_dict))

        for key in config_dict['model_params']:
            self.log.info(key)
            self.assertIsNotNone(key)

    def test_merge_params(self):
        """Checks how parameters passed to TEST_FLAGS interact with
        parameters from yaml files. Expected behavior is that any
        params in TEST_FLAGS will override those from files, but that
        all values from file will be used if not explicitly passed to
        TEST_FLAGS.
        """

        # ==============================================================
        # Easy tests.
        # ==============================================================

        # Change model in flags and ensure merged config uses that model.
        TEST_FLAGS.model = "chatbot.ChatBot"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['model'], 'chatbot.ChatBot')

        # Also ensure that switching back works too.
        TEST_FLAGS.model = "chatbot.DynamicBot"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['model'], 'chatbot.DynamicBot')

        # Do the same for changing the dataset.
        TEST_FLAGS.dataset = "data.TestData"
        config = io_utils.parse_config(TEST_FLAGS)
        self.assertEqual(config['dataset'], 'data.TestData')

        # ==============================================================
        # Medium tests.
        # ==============================================================

        # Ensure recursive merging works.
        reset_flags(TEST_FLAGS)
        test_params = {'batch_size': 123, 'dropout_prob': 0.8}
        TEST_FLAGS.model_params = str(test_params)
        config = io_utils.parse_config(TEST_FLAGS)
        print("Config:\n", config)
        self.assertEqual(config['model'], self.test_config['model'])
        self.assertEqual(config['dataset'], self.test_config['dataset'])
        self.assertNotEqual(config['model_params'], self.test_config['model_params'])

        # Assert order of preference (in assignments) is as expected.
        for p in DEFAULT_FULL_CONFIG['model_params'].keys():
            if p in test_params.keys():
                self.assertEqual(config['model_params'][p], test_params[p])
            elif p in self.test_config['model_params'].keys():
                self.assertEqual(config['model_params'][p],
                                 self.test_config['model_params'][p])
            else:
                self.assertEqual(config['model_params'][p],
                                 DEFAULT_FULL_CONFIG['model_params'][p])


    def test_save_params(self):

        test_params = {'batch_size': 123, 'dropout_prob': 0.8}
        TEST_FLAGS.model_params = str(test_params)
        config = io_utils.parse_config(TEST_FLAGS)
        with open('macros/test_save_false_flow.yml', 'w') as f:
            # Setting flow style False makes it human-friendly (pretty).
            yaml.dump(config, f, default_flow_style=False)




if __name__ == '__main__':
    unittest.main()

