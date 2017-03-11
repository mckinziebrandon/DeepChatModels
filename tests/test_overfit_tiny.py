from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import tensorflow as tf
sys.path.insert(0, os.path.abspath('..'))
from utils import *
from chatbot import DynamicBot
from data import TestData

# ==============================================================================================================================
# Goal: Make sure models can overfit and essentially memorize the small TestData set. This is
# a suggested debugging technique in the textbook "Deep Learning" by Goodfellow et al.
#
# ...Yay. It worked. Good robot.
# ==============================================================================================================================

if __name__ == "__main__":

    chat_session = True
    state_size = 1024
    embed_size = 64
    learning_rate = 0.02
    batch_size = 4

    # All datasets follow the same API, found in data/_dataset.py
    print("Setting up dataset.")
    dataset = TestData()

    # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     ckpt_dir="out",
                     batch_size=batch_size,
                     state_size=state_size,
                     embed_size=embed_size,
                     learning_rate=learning_rate,
                     lr_decay=0.999,
                     steps_per_ckpt=12,
                     is_chatting=chat_session)

    # Don't forget to compile!
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=10.0, reset=False)

    if not chat_session:
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset.train_data, dataset.valid_data, nb_epoch=1000)
    else:
        print("Initiating chat session")
        bot.decode()


