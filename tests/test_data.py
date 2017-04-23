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
import chatbot
import data
from chatbot.globals import DEFAULT_FULL_CONFIG
dir = os.path.dirname(os.path.realpath(__file__))

# Allow user to override config values with command-line args.
# All test_flags with default as None are not accessed unless set.
test_flags = tf.app.flags
test_flags.DEFINE_string("config", "macros/test_config.yml", "path to config (.yml) file.")
test_flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
test_flags.DEFINE_string("model_params", "{}", "")
test_flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,Reddit}.")
test_flags.DEFINE_string("dataset_params", "{}", "")
TEST_FLAGS = test_flags.FLAGS
KEYS = ['model', 'model_params', 'dataset', 'dataset_params']


def reset_flags(flags):
    """Reset flags to the default values."""
    for name in KEYS:
        setattr(flags, name, '{}')


class TestData(unittest.TestCase):
    """Tests for the datsets."""

    def setUp(self):
        self.supported_datasets = ['Reddit', 'Ubuntu', 'Cornell']

    def test_basic(self):
        """Instantiate all supported datasets and check they satisfy basic conditions."""
        dataset_params = {'vocab_size': 40000,
                          'max_seq_len': 10}

        for dataset_name in self.supported_datasets:

            dataset_class = getattr(data, dataset_name)
            # User must specify data_dir, which we have not done yet.
            self.assertRaises(KeyError, dataset_class(dataset_params))

            dataset_params['data_dir'] = '/home/brandon/Datasets/' + dataset_name.lower()
            dataset = dataset_class(dataset_params)

            # Ensure all parms in DEFAULT_FULL_CONFIG['dataset_params'] are specified.
            for default_key in DEFAULT_FULL_CONFIG['dataset_params']:
                self.assertIsNotNone(getattr(dataset, default_key))

            # Check that all dataset properties exist.
            self.assertIsNotNone(dataset.name)
            self.assertIsNotNone(dataset.word_to_idx)
            self.assertIsNotNone(dataset.idx_to_word)
            self.assertIsNotNone(dataset.vocab_size)
            self.assertIsNotNone(dataset.max_seq_len)

            # Check that the properties satisfy basic expectations.
            self.assertEqual(len(dataset.word_to_idx), len(dataset.idx_to_word))
            self.assertEqual(len(dataset.word_to_idx), dataset.vocab_size)
            self.assertEqual(len(dataset.idx_to_word), dataset.vocab_size)




if __name__ == '__main__':
    unittest.main()
