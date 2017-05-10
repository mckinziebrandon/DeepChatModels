"""Trial runs on DynamicBot with the TestData Dataset."""

import time
import logging
import unittest

import numpy as np
import tensorflow as tf
import pydoc
from pydoc import locate

import data
import chatbot
from utils import io_utils, bot_freezer
from tests.utils import *


class TestDynamicModels(unittest.TestCase):

    def setUp(self):
        tf.logging.set_verbosity('ERROR')

    def test_create_bot(self):
        """Ensure bot constructor is error-free."""
        logging.info("Creating bot . . . ")
        bot = create_bot()
        self.assertIsInstance(bot, chatbot.DynamicBot)

    def test_save_bot(self):
        """Ensure we can save to bot ckpt dir."""
        bot = create_bot()
        self.assertIsInstance(bot, chatbot.DynamicBot)

    def test_save_bot(self):
        """Ensure teardown operations are working."""
        bot = create_bot()
        self.assertIsInstance(bot, chatbot.DynamicBot)
        logging.info("Closing bot . . . ")
        bot.close()

    def test_train(self):
        """Simulate a brief training session."""
        flags = TEST_FLAGS
        flags = flags._replace(model_params=dict(
            **flags.model_params,
            reset_model=True,
            steps_per_ckpt=10))
        bot = create_bot(flags)
        self._quick_train(bot)

    def test_base_methods(self):
        """Call each method in chatbot._models.Model, checking for errors."""
        bot = create_bot()
        logging.info('Calling bot.save() . . . ')
        bot.save()
        logging.info('Calling bot.freeze() . . . ')
        bot.freeze()
        logging.info('Calling bot.close() . . . ')
        bot.close()

    def test_manual_freeze(self):
        """Make sure we can freeze the bot, unfreeze, and still chat."""

        # ================================================
        # 1. Create & train bot.
        # ================================================
        flags = TEST_FLAGS
        flags = flags._replace(model_params=dict(
            ckpt_dir=os.path.join(TEST_DIR, 'out'),
            reset_model=True,
            steps_per_ckpt=20,
            max_steps=40))
        bot = create_bot(flags)
        self.assertEqual(bot.reset_model, True)
        # Simulate small train sesh on bot.
        bot.train()

        # ================================================
        # 2. Recreate a chattable bot.
        # ================================================
        # Recreate bot from scratch with decode set to true.
        logging.info("Resetting default graph . . . ")
        tf.reset_default_graph()
        flags = flags._replace(model_params={
            **flags.model_params,
            'reset_model': False,
            'decode': True,
            'max_steps': 100,
            'steps_per_ckpt': 50})
        self.assertTrue(flags.model_params.get('decode'))
        bot = create_bot(flags)
        self.assertTrue(bot.is_chatting)
        self.assertTrue(bot.decode)

        print("Testing quick chat sesh . . . ")
        config = io_utils.parse_config(flags=flags)
        dataset_class = pydoc.locate(config['dataset']) \
                        or getattr(data, config['dataset'])
        dataset = dataset_class(config['dataset_params'])
        test_input = "How's it going?"
        encoder_inputs = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(test_input),
            dataset.word_to_idx)
        encoder_inputs = np.array([encoder_inputs[::-1]])
        bot.pipeline._feed_dict = {
            bot.pipeline.user_input: encoder_inputs}

        # Get output sentence from the chatbot.
        _, _, response = bot.step(forward_only=True)
        print("Robot:", dataset.as_words(response[0][:-1]))

        # ================================================
        # 3. Freeze the chattable bot.
        # ================================================
        logging.info("Calling bot.freeze() . . . ")
        bot.freeze()

        # ================================================
        # 4. Try to unfreeze and use it.
        # ================================================
        logging.info("Resetting default graph . . . ")
        tf.reset_default_graph()
        logging.info("Importing frozen graph into default . . . ")
        frozen_graph = bot_freezer.load_graph(bot.ckpt_dir)

        logging.info("Extracting input/output tensors.")
        tensors, frozen_graph = bot_freezer.unfreeze_bot(bot.ckpt_dir)
        self.assertIsNotNone(tensors['inputs'])
        self.assertIsNotNone(tensors['outputs'])

        with tf.Session(graph=frozen_graph) as sess:
            raw_input = "How's it going?"
            encoder_inputs  = io_utils.sentence_to_token_ids(
                tf.compat.as_bytes(raw_input),
                dataset.word_to_idx)
            encoder_inputs = np.array([encoder_inputs[::-1]])
            feed_dict = {tensors['inputs'].name: encoder_inputs}
            response = sess.run(tensors['outputs'], feed_dict=feed_dict)
            logging.info('Reponse: %s', response)


    def test_memorize(self):
        """Train a bot to memorize (overfit) the small test data, and 
        show its responses to all train inputs when done.
        """

        flags = TEST_FLAGS
        flags = flags._replace(model_params=dict(
            ckpt_dir='out/test_data',
            reset_model=True,
            steps_per_ckpt=300,
            state_size=128,
            embed_size=32,
            max_steps=300))
        flags = flags._replace(dataset_params=dict(
            max_seq_len=20,
            data_dir=TEST_DATA_DIR))
        print('TEST_FLAGS', flags.dataset)
        bot, dataset = create_bot(flags=flags, return_dataset=True)
        bot.train()

        # Recreate bot (its session is automatically closed after training).
        flags = flags._replace(model_params={
            **flags.model_params,
            'reset_model': False,
            'decode': True})
        bot, dataset = create_bot(flags, return_dataset=True)

        for inp_sent, resp_sent in dataset.pairs_generator():
            print('\nHuman:', inp_sent)
            response = bot.respond(inp_sent)
            if response == resp_sent:
                print('Robot: %s\nCorrect!' % response)
            else:
                print('Robot: %s\nExpected: %s' % (
                    response, resp_sent))


    def _quick_train(self, bot, num_iter=10):
        """Quickly train manually on some test data."""
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=bot.sess, coord=coord)
        for _ in range(num_iter):
            bot.step()
        summaries, loss, _ = bot.step()
        bot.save(summaries=summaries)
        coord.request_stop()
        coord.join(threads)

