#!/usr/bin/env python3

"""main.py: Train and/or chat with a bot. (work in progress).

Typical use cases:
    1.  Train a model specified by yaml config file, located at
        path_to/my_config.yml, where paths are relative to project root:
            ./main.py --config path_to/my_config.yml

    2.  Train using mix of yaml config and cmd-line args, with
        command-line args taking precedence over any values.
            ./main.py \
                --config path_to/my_config.yml \
                --model_params "{'batch_size': 32, 'optimizer': 'RMSProp'}"

    3.  Load a pretrained model that was saved in path_to/pretrained_dir,
        which is assumed to be relative to the project root.
            ./main.py --pretrained_dir path_to/pretrained_dir

"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
# Meaning of values:
#   1: INFO messages are not printed.
#   2: INFO, WARNING messages are not printed.
# I'm temporarily making the default '2' since the TF master
# branch (as of May 6) is spewing warnings that are clearly
# due to bugs on their side.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import data
import chatbot
import logging
import tensorflow as tf
from pydoc import locate
from utils import io_utils

# =============================================================================
# FLAGS: Command line argument parser from TensorFlow.
# =============================================================================

flags = tf.app.flags
flags.DEFINE_string(
    flag_name="pretrained_dir",
    default_value=None,
    docstring="relative path to a pretrained model directory."
              "It is assumed that the model is one from this repository, and "
              " thus has certain files that are generated after any training"
              " session (TL;DR: any ckpt_dir you've trained previously).")
flags.DEFINE_string(
    flag_name="config",
    default_value=None,
    docstring="relative path to a valid yaml config file."
              " For example: configs/example_cornell.yml")
flags.DEFINE_string(
    flag_name="debug",
    default_value=False,
    docstring="If true, increases output verbosity (log levels).")
flags.DEFINE_string(
    flag_name="model",
    default_value="{}",
    docstring="Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
flags.DEFINE_string(
    flag_name="model_params",
    default_value="{}",
    docstring="Configuration dictionary, with supported keys specified by"
              " those in chatbot.globals.py.")
flags.DEFINE_string(
    flag_name="dataset",
    default_value="{}",
    docstring="Name (capitalized) of dataset to use."
              " Options: [data.]{Cornell,Ubuntu,Reddit}."
              " - Legend: [optional] {Pick,One,Of,These}.")
flags.DEFINE_string(
    flag_name="dataset_params",
    default_value="{}",
    docstring="Configuration dictionary, with supported keys specified by"
              " those in chatbot.globals.py.")
FLAGS = flags.FLAGS


def start_training(dataset, bot):
    """Train bot. 
    
    Will expand this function later to aid interactivity/updates.
    """
    print("Training bot. CTRL-C to stop training.")
    bot.train(dataset)


def start_chatting(bot):
    """Talk to bot. 
    
    Will re-add teacher mode soon. Old implementation in _decode.py."""
    print("Initiating chat session.")
    print("Your bot has a temperature of %.2f." % bot.temperature, end=" ")
    if bot.temperature < 0.1:
        print("Not very adventurous, are we?")
    elif bot.temperature < 0.7:
        print("This should be interesting . . . ")
    else:
        print("Enjoy your gibberish!")
    bot.chat()


def main(argv):

    if FLAGS.debug:
        # Setting to '0': all tensorflow messages are logged.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        logging.basicConfig(level=logging.INFO)

    # Extract the merged configs/dictionaries.
    config = io_utils.parse_config(flags=FLAGS)
    if config['model_params']['decode'] and config['model_params']['reset_model']:
        print("Woops! You passed {decode: True, reset_model: True}." 
              " You can't chat with a reset bot! I'll set reset to False.")
        config['model_params']['reset_model'] = False

    # If loading from pretrained, double-check that certain values are correct.
    # (This is not something a user need worry about -- done automatically)
    if FLAGS.pretrained_dir is not None:
        assert config['model_params']['decode'] \
               and not config['model_params']['reset_model']

    # Print out any non-default parameters given by user, so as to reassure
    # them that everything is set up properly.
    io_utils.print_non_defaults(config)

    print("Setting up %s dataset." % config['dataset'])
    dataset_class = locate(config['dataset']) or getattr(data, config['dataset'])
    dataset = dataset_class(config['dataset_params'])
    print("Creating", config['model'], ". . . ")
    bot_class = locate(config['model']) or getattr(chatbot, config['model'])
    bot = bot_class(dataset, config)

    if not config['model_params']['decode']:
        start_training(dataset, bot)
    else:
        start_chatting(bot)

if __name__ == "__main__":
    tf.logging.set_verbosity('ERROR')
    tf.app.run()

