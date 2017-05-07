"""Utility functions used by test modules."""

import logging
import data
import chatbot

from pydoc import locate
from utils import io_utils
import tensorflow as tf

from main import FLAGS as TEST_FLAGS
TEST_FLAGS.config = "configs/test_config.yml"
logging.basicConfig(level=logging.INFO)


def create_bot(flags=TEST_FLAGS, return_dataset=False):
    """Chatbot factory: Creates and returns a fresh bot. Nice for 
    testing specific methods quickly.
    """

    # Wipe the graph and update config if needed.
    tf.reset_default_graph()
    config = io_utils.parse_config(flags=flags)
    io_utils.print_non_defaults(config)

    # Instantiate a new dataset.
    print("Setting up", config['dataset'], "dataset.")
    dataset_class = locate(config['dataset']) \
                    or getattr(data, config['dataset'])
    dataset = dataset_class(config['dataset_params'])

    # Instantiate a new chatbot.
    print("Creating", config['model'], ". . . ")
    bot_class = locate(config['model']) or getattr(chatbot, config['model'])
    bot = bot_class(dataset, config)

    if return_dataset:
        return bot, dataset
    else:
        return bot


