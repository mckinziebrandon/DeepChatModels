import os
import tensorflow as tf
import unittest
import logging

import sys
sys.path.append("..")
from chatbot import ChatBot, SimpleBot
from data import Cornell, Ubuntu, TestData

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"

flags = tf.app.flags
# String test_flags -- directories and dataset name(s).
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", "out/logs", ".")
flags.DEFINE_string("dataset", "cornell", "Dataset to use. 'ubuntu', 'cornell', or 'wmt'.")
# Boolean test_flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, initiates chat session.")
# Integer test_flags.
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("state_size", 512, "Number of units in the RNN cell.")
flags.DEFINE_integer("embed_size", 64, "Size of word embedding dimension.")
flags.DEFINE_integer("nb_epoch", 10, "Number of epochs over full train set to run.")
flags.DEFINE_integer("num_layers", 3, "Num layers in underlying MultiRNNCell.")
# Float test_flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
flags.DEFINE_float("temperature", 0.01, "Sampling temperature.")
flags.DEFINE_float("dropout_prob", 0.5, "Dropout rate before each layer.")
FLAGS = flags.FLAGS

class TestLegacyModels(unittest.TestCase):
    """Test behavior of tf.contrib.rnn after migrating to r1.0."""

    def setUp(self):
        self.seq_len = 20
        self.dataset = TestData()
        self.batch_size = 2
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestLegacyModels')

    def test_simple_bot_train(self):
        """Test basic functionality of SimpleBot remains up-to-date with _models."""
        bot = SimpleBot(self.dataset.name, batch_size=self.batch_size)
        bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)
        self.log.info("\n [SimpleBot] Initiating training session.")
        bot.train(self.dataset)

    def test_chat_bot_train(self):
        """Test basic functionality of SimpleBot remains up-to-date with _models."""
        buckets = [(10, 20)]
        bot = ChatBot(buckets, self.dataset.name, batch_size=self.batch_size)
        bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)
        self.log.info("\n [SimpleBot] Initiating training session.")
        bot.train(self.dataset)


if __name__ == '__main__':
    unittest.main()
