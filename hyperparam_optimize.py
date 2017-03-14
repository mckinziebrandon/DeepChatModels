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
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("nb_epoch", 2, "Number of epochs over full train set to run.")
flags.DEFINE_float("max_gradient", 10.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    # All datasets follow the same API, found in data/_dataset.py
    print("Setting up dataset.")
    dataset = Cornell(FLAGS.vocab_size)

    embed_size = 32
    learning_rate = 0.5
    state_size = 128


    # Switching back to grid search, it appears better when the parameter set size < 5.
    hyperparams = []
    for n_layer in [2, 4]:
            for dr_prob in [0.2, 0.5, 0.8]:
                hyperparams.append([n_layer, dr_prob])

    print('\n\n')

    for num_layers, dropout_prob in hyperparams:
        print("=================== NEW RUN ===================")
        print("learning rate:", learning_rate)
        print("state size:", state_size)
        print("num_layers:", num_layers)
        print("dropout_prob:", dropout_prob)
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
                         num_layers=num_layers,
                         dropout_prob=dropout_prob,
                         is_chatting=False)
        # Don't forget to compile!
        print("Compiling DynamicBot.")
        bot.compile(max_gradient=FLAGS.max_gradient, reset=True)
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset, nb_epoch=FLAGS.nb_epoch, searching_hyperparams=True)

        tf.reset_default_graph()



