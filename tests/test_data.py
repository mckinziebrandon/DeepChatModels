import logging
import pdb
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
from pydoc import locate
import chatbot
from utils import io_utils
import data
from chatbot.globals import DEFAULT_FULL_CONFIG
dir = os.path.dirname(os.path.realpath(__file__))
from tests.utils import *


class TestData(unittest.TestCase):
    """Tests for the datsets."""

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        tf.logging.set_verbosity('ERROR')
        self.supported_datasets = ['Reddit', 'Ubuntu', 'Cornell']
        self.default_flags = {
            'pretrained_dir': TEST_FLAGS.pretrained_dir,
            'config': TEST_FLAGS.config,
            'model': TEST_FLAGS.model,
            'debug': TEST_FLAGS.debug}

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

    def test_cornell(self):
        """Train a bot on cornell and display responses when given
        training data as input -- a sanity check that the data is clean.
        """

        flags = Flags(
            model_params=dict(
                ckpt_dir='out/tests/test_cornell',
                reset_model=True,
                steps_per_ckpt=50,
                base_cell='GRUCell',
                num_layers=1,
                state_size=128,
                embed_size=64,
                max_steps=50),
            dataset_params=dict(
                vocab_size=50000,
                max_seq_len=8,
                data_dir='/home/brandon/Datasets/cornell'),
            dataset='Cornell',
            **self.default_flags)

        bot, dataset = create_bot(flags=flags, return_dataset=True)
        bot.train()

        del bot

        # Recreate bot (its session is automatically closed after training).
        flags.model_params['reset_model'] = False
        flags.model_params['decode'] = True
        bot, dataset = create_bot(flags, return_dataset=True)

        for inp_sent, resp_sent in dataset.pairs_generator(100):
            print('\nHuman:', inp_sent)
            response = bot.respond(inp_sent)
            if response == resp_sent:
                print('Robot: %s\nCorrect!' % response)
            else:
                print('Robot: %s\nExpected: %s' % (
                    response, resp_sent))


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    unittest.main()
