#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
from chatbot import DynamicBot, ChatBot, SimpleBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils
import tensorflow as tf
import sys, getopt
from pydoc import locate

# Allow user to override config values with command-line args.
# All flags with default as None are not accessed unless set.
flags = tf.app.flags
flags.DEFINE_string("config", "configs/default.yml", "path to config (.yml) file. Defaults to DynamicBot on Cornell.")
flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
flags.DEFINE_string("model_params", "{}", "")
flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
flags.DEFINE_string("dataset_params", "{}", "")
FLAGS = flags.FLAGS


def start_training(dataset, bot):
    """Train bot. Will expand this function later to aid interactivity/updates. Maybe."""
    print("Training bot. CTRL-C to stop training.")
    bot.train(dataset)


def start_chatting(bot):
    """Talk to bot. Will add teacher mode soon. Old implementation in _decode.py."""
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

    # Extract merge configs/dictionaries.
    config = io_utils.parse_config(FLAGS)

    print("Setting up %s dataset." % config['dataset'])
    dataset = locate(config['dataset'])(config['dataset_params'])
    print("Creating", config['model'], ". . . ")
    bot = locate(config['model'])(dataset, config['model_params'])

    exit()

    if not config['model_params']['decode']:
        start_training(dataset, bot)
    else:
        start_chatting(bot)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.app.run()

