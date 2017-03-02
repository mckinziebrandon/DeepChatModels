import os
import unittest
from utils import Config
import chatbot
import tensorflow as tf

#DEFAULT_DATA_DIR="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
DEFAULT_DATA_DIR= "/home/brandon/terabyte/Datasets/wmt"

CWD = os.getcwd()
TEST_CKPT_DIR = os.path.join(CWD, "out")

flags = tf.app.flags
flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Directory containing the data files.")
flags.DEFINE_string("ckpt_dir", TEST_CKPT_DIR, "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("reset_model", True, "wipe output directory; new params")

flags.DEFINE_string("chunk_size", 10000, "")
flags.DEFINE_integer("max_train_samples", int(22e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")

flags.DEFINE_string("data_name", "wmt", "For now, either 'ubuntu' or 'wmt'.")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 256, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
FLAGS = flags.FLAGS

class TestChatbot(unittest.TestCase):

    def test_train(self):
        config = Config(FLAGS)
        buckets = [(5, 10)]
        bot = chatbot.Chatbot(buckets,
                              layer_size=FLAGS.layer_size,
                              num_layers=FLAGS.num_layers,
                              debug_mode=True)
        bot.train(config)
