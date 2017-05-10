"""Utility functions used by test modules."""

import logging
import data
import chatbot

import os
from pydoc import locate
import pdb
from utils import io_utils
import tensorflow as tf
from chatbot.globals import DEFAULT_FULL_CONFIG
from collections import namedtuple

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'test_data')
TEST_CONFIG_PATH = os.path.join(TEST_DIR, 'test_config.yml')
logging.basicConfig(level=logging.INFO)

_flag_names = ["pretrained_dir",
               "config",
               "debug",
               "model",
               "model_params",
               "dataset",
               "dataset_params"]
Flags = namedtuple('Flags', _flag_names)
TEST_FLAGS = Flags(pretrained_dir=None,
                   config=TEST_CONFIG_PATH,
                   debug=True,
                   model='{}',
                   dataset='{}',
                   model_params={'ckpt_dir': os.path.join(TEST_DIR, 'out')},
                   dataset_params={'data_dir': TEST_DATA_DIR})


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


def update_config(config, **kwargs):
    new_config = {}
    for key in DEFAULT_FULL_CONFIG:
        for new_key in kwargs:
            if new_key in DEFAULT_FULL_CONFIG[key]:
                if new_config.get(key) is None:
                    new_config[key] = {}
                new_config[key][new_key] = kwargs[new_key]
            elif new_key == key:
                new_config[new_key] = kwargs[new_key]
    return {**config, **new_config}

