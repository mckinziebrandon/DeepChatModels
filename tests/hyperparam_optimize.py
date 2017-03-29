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
import os

flags = tf.app.flags
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("search_choice", "optimizers", "Whether to search 'optimizers' or 'hparams'.")
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("nb_epoch", 2, "Number of epochs over full train set to run.")
flags.DEFINE_float("max_gradient", 10.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

# 'Fixed' params in that they aren't iterated over in hparam search.
state_size = 256
steps_per_ckpt = 500
default_dict = {'dropout_prob': 0.2, 'batch_size': 64,
                'learning_rate': 0.4, 'num_layers': 2}

print("Setting up cornell dataset.")
DATA_DIR = '/home/brandon/terabyte/Datasets/cornell'
dataset = Cornell(DATA_DIR, FLAGS.vocab_size)


def scalar_hyperparam_search(hyperparams):
    """Train a bot for TEST_FLAGS.nb_epoch with each set of hyperparameters in input list.

    Args:
        list of dictionaries from name -> float/int hyperparemeters to initialize the bot.
        Supported keys: dropout_prob, batch_size, learning_rate, num_layers.
    """

    # If user dataset in text format, reformat it into tensorflow protobuf.
    dataset.convert_to_tf_records('train')
    dataset.convert_to_tf_records('valid')
    for i in range(len(hyperparams)):

        hparams = hyperparams[i]
        # Set any unspecified values to defaults.
        for key in default_dict.keys():
            if key not in hyperparams[i]:
                hparams[key] = default_dict[key]

        print("=================== NEW RUN ===================")
        print("learning rate:", hparams['learning_rate'])
        print("state size:", state_size)
        print("num_layers:", hparams['num_layers'])
        print("dropout_prob:", hparams['dropout_prob'])
        print("===============================================\n")

        hparam_string = 'lr_%d_nlay_%d_drop_%d' % (int(1e2 * hparams['learning_rate']),
                                                   hparams['num_layers'],
                                                   int(1e2 * hparams['dropout_prob']))
        ckpt_dir = os.path.join(FLAGS.ckpt_dir, dataset.name, hparam_string)

        bot = DynamicBot(dataset,
                         ckpt_dir=ckpt_dir,
                         batch_size=hparams['batch_size'],
                         learning_rate=hparams['learning_rate'],
                         num_layers=hparams['num_layers'],
                         dropout_prob=hparams['dropout_prob'],
                         state_size=state_size,
                         steps_per_ckpt=steps_per_ckpt)

        # Don't forget to compile!
        print("Compiling DynamicBot.")
        bot.compile(max_gradient=FLAGS.max_gradient, reset=True)
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset, nb_epoch=FLAGS.nb_epoch)
        tf.reset_default_graph()


def optimizer_search(optimizers):
    """Train the bot for TEST_FLAGS.nb_epoch for each optimizer in list.

    Args:
        list of tuples: (name, optimizer) inheriting from tf.train.Optimizer.
    """

    for optim_name, optimizer in optimizers:
        ckpt_dir = os.path.join(FLAGS.ckpt_dir, dataset.name, optim_name)
        bot = DynamicBot(dataset,
                         ckpt_dir=ckpt_dir,
                         batch_size=default_dict['batch_size'],
                         learning_rate=default_dict['learning_rate'],
                         num_layers=default_dict['num_layers'],
                         dropout_prob=default_dict['dropout_prob'])

        print("Compiling using %s as optimizer." % optim_name)
        bot.compile(optimizer=optimizer,
                    max_gradient=FLAGS.max_gradient,
                    reset=True)
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset, nb_epoch=FLAGS.nb_epoch)
        tf.reset_default_graph()

if __name__ == "__main__":

    if FLAGS.search_choice == 'hparams':
        hyperparams = []
        for lr in [0.1, 0.3, 0.5]:
            for drop_prob in [0.2, 0.5]:
                hyperparams.append({'learning_rate': lr, 'dropout_prob': drop_prob})
        scalar_hyperparam_search(hyperparams)
    elif FLAGS.search_choice == 'optimizers':
        optimizers = [('adagrad', tf.train.AdagradOptimizer(default_dict['learning_rate'])),
                      ('rmsprop', tf.train.RMSPropOptimizer(0.1 * default_dict['learning_rate']))]
        optimizer_search(optimizers)
    else:
        print("You did not pass a valid search choice. I am confused and terminating the program.")


