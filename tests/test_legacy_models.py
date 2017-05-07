import os
import tensorflow as tf
import unittest
import logging

import sys
from utils import io_utils
import data
import chatbot

from tests.utils import TEST_FLAGS

class TestLegacyModels(unittest.TestCase):
    """Test behavior of tf.contrib.rnn after migrating to r1.0."""

    def setUp(self):
        self.seq_len = 20
        self.config = io_utils.parse_config(flags=TEST_FLAGS)
        self.dataset = data.TestData(self.config['dataset_params'])
        self.batch_size = 2
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestLegacyModels')

    def test_create(self):
        """Test basic functionality of SimpleBot remains up-to-date with _models."""
        simple_bot = chatbot.SimpleBot(
            dataset=self.dataset,
            params=self.config)
        self.assertIsInstance(simple_bot, chatbot.SimpleBot)

        chat_bot = chatbot.ChatBot(
            buckets=[(10, 10)],
            dataset=self.dataset,
            params=self.config)
        self.assertIsInstance(chat_bot, chatbot.ChatBot)

    def test_compile(self):
        """Test basic functionality of SimpleBot remains up-to-date with _models."""
        buckets = [(10, 20)]

        # SimpleBot
        logging.info("Creating/compiling SimpleBot . . . ")
        bot = chatbot.SimpleBot(
            dataset=self.dataset,
            params=self.config)
        bot.compile()

        # ChatBot
        logging.info("Creating/compiling ChatBot . . . ")
        bot = chatbot.ChatBot(
            buckets=buckets,
            dataset=self.dataset,
            params=self.config)
        bot.compile()


if __name__ == '__main__':
    unittest.main()
