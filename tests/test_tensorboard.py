import os
import numpy as np
import unittest
from utils import Config
import chatbot
import time
import tensorflow as tf
from utils import data_utils

DEFAULT_DATA_DIR="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
#DEFAULT_DATA_DIR= "/home/brandon/terabyte/Datasets/wmt"

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("data_name", "ubuntu", "For now, either 'ubuntu' or 'wmt'.")
flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Directory containing the data files.")
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", "out/logs", "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
# Integer flags -- First three only need custom values if you're especially worried about RAM.
flags.DEFINE_integer("max_train_samples", int(22e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("chunk_size", int(1e6), "")
flags.DEFINE_integer("steps_per_ckpt", 50, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 128, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
# Float flags -- training hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.97, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

class TestTensorboard(unittest.TestCase):

    # setUp is called by unittest before any/all test(s).
    def setUp(self):
        self.config = Config(FLAGS)
        buckets = [(5, 10)]
        self.bot = chatbot.Chatbot(buckets,
                              layer_size=FLAGS.layer_size,
                              num_layers=FLAGS.num_layers,
                              debug_mode=True)

    def test_merged(self):
        """Ensure chatbot.merged gets properly initialized."""
        config = Config(FLAGS)
        buckets = [(5, 10)]
        bot = chatbot.Chatbot(buckets,
                              layer_size=FLAGS.layer_size,
                              num_layers=FLAGS.num_layers,
                              debug_mode=True)
        self.assertIsNotNone(bot.merged)

    def test_train_step(self):
        """Check for expected outputs of chatbot.step."""

        def _get_data_distribution(train_set, buckets):
            train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
            train_total_size   = float(sum(train_bucket_sizes))
            return [sum(train_bucket_sizes[:i + 1]) / train_total_size
                             for i in range(len(train_bucket_sizes))]

        self.bot.sess = self.bot._create_session()
        self.bot._setup_parameters(self.config)

        with self.bot.sess as sess:
            print ("Reading development and training data (limit: %d)." % self.config.max_train_samples)
            train_set, dev_set = data_utils.read_data(self.config.dataset,
                                                      self.bot.buckets,
                                                      max_train_data_size=self.config.chunk_size)
            train_buckets_scale = _get_data_distribution(train_set, self.bot.buckets)
            step_time, loss = 0.0, 0.0
            previous_losses = []
            for i_step in range(5):
                rand = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = self.bot.get_batch(train_set, bucket_id)
                step_returns = self.bot.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,False)

                merged, gradient_norms, losses, _ = step_returns
                self.bot.train_writer.add_summary(merged, i_step)
                step_time += (time.time() - start_time) / self.config.steps_per_ckpt
                loss      += losses / self.config.steps_per_ckpt


