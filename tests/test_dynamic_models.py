"""Run trial run on DynamicBot with the TestData Dataset."""
import os
import time
import numpy as np
import logging
import unittest
import sys
sys.path.append("..")
from data import *
from chatbot import DynamicBot
from utils import io_utils


class TestDynamicModels(unittest.TestCase):


    def test_chat(self):
        """Feed the training sentences to the bot during conversation.
        It should respond somewhat predictably on these for now.
        """

        data_dir = '/home/brandon/terabyte/Datasets/test_data'
        dataset = TestData(data_dir)
        dataset.convert_to_tf_records('train')
        dataset.convert_to_tf_records('valid')

        print("Should I train first?")
        should_train = io_utils.get_sentence()
        is_chatting = False if should_train == 'y' else True
        print("is chatting is ", is_chatting)

        bot = DynamicBot(dataset,
                         batch_size=4,
                         learning_rate=0.05,
                         state_size=1024,
                         embed_size=64,
                         num_layers=3,
                         is_chatting=is_chatting)
        bot.compile(reset=(not is_chatting))
        if not is_chatting:
            bot.train(dataset)
        else:
            sentence_generator = dataset.sentence_generator()
            try:
                while True:
                    sentence = next(sentence_generator)
                    print("Human:\t", sentence)
                    print("Bot:  \t", bot(sentence))
                    print()
                    time.sleep(1)
            except (KeyboardInterrupt, StopIteration):
                print('Bleep bloop. Goodbye.')


