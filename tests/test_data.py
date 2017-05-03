import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
from utils import io_utils
import data
from chatbot.globals import DEFAULT_FULL_CONFIG
dir = os.path.dirname(os.path.realpath(__file__))

# Allow user to override config values with command-line args.
# All test_flags with default as None are not accessed unless set.
from main import FLAGS as TEST_FLAGS
TEST_CONFIG_PATH = "configs/test_config.yml"
TEST_FLAGS.config = TEST_CONFIG_PATH
logging.basicConfig(level=logging.INFO)


class TestData(unittest.TestCase):
    """Tests for the datsets."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.supported_datasets = ['Reddit', 'Ubuntu', 'Cornell']

    def test_basic(self):
        """Instantiate all supported datasets and check they satisfy basic conditions.
        
        THIS MAY TAKE A LONG TIME TO COMPLETE. Since we are testing that the 
        supported datasets can be instantiated successfully, it necessarily 
        means that the data must exist in proper format. Since the program
        will generate the proper format(s) if not found, this will take 
        about 15 minutes if run from a completely fresh setup.
        
        Otherwise, a few seconds. :)
        """

        if os.getenv('DATA') is None \
            and not os.path.exists('/home/brandon/Datasets'):
            print('To run this test, please enter the path to your datasets: ')
            data_dir = input()
        else:
            data_dir = '/home/brandon/Datasets'

        for dataset_name in self.supported_datasets:
            logging.info('Testing %s', dataset_name)

            incomplete_params = {
                'vocab_size': 40000,
                'max_seq_len': 10}
            self.assertIsNotNone(incomplete_params)
            dataset_class = getattr(data, dataset_name)
            # User must specify data_dir, which we have not done yet.
            self.assertRaises(ValueError, dataset_class, incomplete_params)

            config = io_utils.parse_config(flags=TEST_FLAGS)
            dataset_params = config.get('dataset_params')
            dataset_params['data_dir'] = os.path.join(
                data_dir,
                dataset_name.lower())
            dataset = dataset_class(dataset_params)

            # Ensure all params from DEFAULT_FULL_CONFIG['dataset_params']
            # are set to a value in our dataset object.
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

            incomplete_params.clear()
            dataset_params.clear()


if __name__ == '__main__':
    unittest.main()
