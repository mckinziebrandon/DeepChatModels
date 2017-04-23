"""Trial runs on DynamicBot with the TestData Dataset."""

import time
import logging
import unittest

import numpy as np
import tensorflow as tf
from pydoc import locate

import data
import chatbot
from utils import io_utils
from utils import bot_freezer

test_flags = tf.app.flags
test_flags.DEFINE_string("config", "configs/test_config.yml", "path to config (.yml) file.")
test_flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
test_flags.DEFINE_string("model_params", "{}", "")
test_flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
test_flags.DEFINE_string("dataset_params", "{}", "")
TEST_FLAGS = test_flags.FLAGS


def get_default_bot(flags=TEST_FLAGS):
    """Creates and returns a fresh bot. Nice for testing specific methods quickly."""
    tf.reset_default_graph()
    config = io_utils.parse_config(flags)
    print("Setting up %s dataset." % config['dataset'])
    dataset_class = locate(config['dataset']) or getattr(data, config['dataset'])
    dataset = dataset_class(config['dataset_params'])
    print("Creating", config['model'], ". . . ")
    bot_class = locate(config['model']) or getattr(chatbot, config['model'])
    bot = bot_class(dataset, config)
    return bot


class TestDynamicModels(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestDynamicModelsLogger')

    def test_train(self):
        flags = TEST_FLAGS
        flags.model_params = (
            "ckpt_dir: out/test_data,"
            "reset_model: True,"
            "steps_per_ckpt: 10")
        bot = get_default_bot(flags)
        bot.train()

    def test_manual_freeze(self):
        """Make sure we can freeze the bot, unfreeze, and still chat."""

        # ================================================
        # 1. Create & train bot.
        # ================================================
        flags = TEST_FLAGS
        flags.model_params = "{ckpt_dir: out/test_data, " \
                             "reset_model: True, " \
                             "steps_per_ckpt: 10}"
        bot = get_default_bot(flags)
        # Simulate small train sesh on bot.
        self._quick_train(bot)

        # ================================================
        # 2. Recreate a chattable bot.
        # ================================================
        # Recreate bot from scratch with decode set to true.
        self.log.info("Resetting default graph . . . ")
        tf.reset_default_graph()
        flags.model_params = "{ckpt_dir: out/test_data, " \
                             "reset_model: False, " \
                             "decode: True," \
                             "steps_per_ckpt: 10}"
        bot = get_default_bot(flags)
        self.assertTrue(bot.is_chatting)
        self.assertTrue(bot.decode)

        print("Testing quick chat sesh . . . ")
        config = io_utils.parse_config(flags)
        dataset = locate(config['dataset'])(config['dataset_params'])
        user_input = io_utils.get_sentence()
        encoder_inputs = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(user_input),
            dataset.word_to_idx)
        encoder_inputs = np.array([encoder_inputs[::-1]])
        bot.pipeline._feed_dict = {
            bot.pipeline.user_input: encoder_inputs}

        # Get output sentence from the chatbot.
        _, _, response = bot.step(forward_only=True)
        # response has shape [1, response_length] and it's last elemeot is EOS_ID. :)
        print("Robot:", dataset.as_words(response[0][:-1]))

        # ================================================
        # 3. Freeze the chattable bot.
        # ================================================
        self.log.info("Calling bot.freeze() . . . ")
        bot.freeze()

        # ================================================
        # 4. Try to unfreeze and use it.
        # ================================================
        self.log.info("Resetting default graph . . . ")
        tf.reset_default_graph()
        self.log.info("Importing frozen graph into default . . . ")
        frozen_graph = bot_freezer.load_graph(bot.ckpt_dir)
        self._print_op_names(frozen_graph)

        self.log.info("Extracting input/output tensors.")
        tensors = bot_freezer.unfreeze_bot(bot.ckpt_dir)
        keys = ['user_input', 'encoder_inputs', 'outputs']
        for k in keys:
            self.assertIsNotNone(tensors[k])

        with tf.Session(graph=frozen_graph) as sess:
            raw_input = io_utils.get_sentence()
            encoder_inputs  = io_utils.sentence_to_token_ids(
                tf.compat.as_bytes(raw_input),
                dataset.word_to_idx)
            encoder_inputs = np.array([encoder_inputs[::-1]])
            feed_dict = {tensors['user_input'].name: encoder_inputs}
            plz = sess.run(tensors['outputs'], feed_dict=feed_dict)

    def _print_op_names(self, g):
        print("List of Graph Ops:")
        for op in g.get_operations():
            print(op.name)

    def _quick_train(self, bot):
        """Quickly train manually on some test data."""
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=bot.sess, coord=coord)
        for _ in range(10):
            bot.step()
        summaries, loss, _ = bot.step()
        bot.save(summaries=summaries)
        coord.request_stop()
        coord.join(threads)

