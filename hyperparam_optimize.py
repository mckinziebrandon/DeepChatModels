#!/usr/bin/env python3
"""Run random search over a set of hyperparameters. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
from chatbot import DynamicBot
from data import Cornell, Ubuntu, TestData
from utils import io_utils

flags = tf.app.flags
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_integer("vocab_size", 10000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("nb_epoch", 2, "Number of epochs over full train set to run.")
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("max_gradient", 10.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    # All datasets follow the same API, found in data/_dataset.py
    print("Setting up dataset.")
    dataset = Cornell(FLAGS.vocab_size)

    # ____ SWITCHING TO GRID SEARCH because I'm tired of getting triplets like (244, 344, 377).___
    # We eventally want to get samples in range 10^-2 and 10^0.
    # log_learning_rates  = np.random.uniform(-2, 0, size=4)
    # We'll do these base-2 (lg).
    # lg_state_sizes     = np.random.uniform(7, 9, size=3)
    # lg_embed_sizes      = np.random.uniform(4, 6, size=3)
    # lg_embed_sizes      = [5, 6]
    hyperparams = []
    for lr in [0.06, 0.2, 0.7]:
        for state in [128, 256, 512]:
            for embed in [32, 64]:
                hyperparams.append([lr, state, embed])

    #for lg_lr in log_learning_rates:
    #    for lg_state in lg_state_sizes:
    #        for lg_embed in lg_embed_sizes:
    #            hyperparams.append([10**lg_lr, int(2**lg_state), int(2**lg_embed)])
    #            print(hyperparams[-1])

    print('\n\n')

    for learning_rate, state_size, embed_size in hyperparams:
        print("=================== NEW RUN ===================")
        print("learning rate:", learning_rate)
        print("state size:", state_size)
        print("embed size:", embed_size)
        print("===============================================\n")

        # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
        print("Creating DynamicBot.")
        bot = DynamicBot(dataset,
                         ckpt_dir=FLAGS.ckpt_dir,
                         batch_size=64,
                         state_size=state_size,
                         embed_size=embed_size,
                         learning_rate=learning_rate,
                         lr_decay=0.99,
                         steps_per_ckpt=200,
                         is_chatting=False)
        # Don't forget to compile!
        print("Compiling DynamicBot.")
        bot.compile(max_gradient=FLAGS.max_gradient, reset=True)
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset.train_data, dataset.valid_data, nb_epoch=FLAGS.nb_epoch)

        tf.reset_default_graph()



