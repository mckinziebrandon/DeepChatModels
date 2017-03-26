#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
import time
import tensorflow as tf
from chatbot import DynamicBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils

flags = tf.app.flags
flags.DEFINE_string("ckpt_dir", "pretrained/reddit", "Location of ckpt files.")
flags.DEFINE_string("data_dir", None, "Directory containing the training/valid data.")
flags.DEFINE_string("dataset", "cornell", "Dataset to use. 'ubuntu', 'cornell', or 'wmt'.")
flags.DEFINE_float("temperature", 0.0, "Sampling temperature.")
FLAGS = flags.FLAGS

DATASET = {'ubuntu': Ubuntu,
           'cornell': Cornell,
           'wmt': WMT,
           'reddit': Reddit,
           'test_data': TestData}

if __name__ == "__main__":

    configs = io_utils.parse_config(FLAGS.config_path)
    try:
        model_name      = configs['model']
        dataset_name    = configs['dataset']
        dataset_params  = configs['dataset_params']
        model_params    = configs['model_params']
    except KeyError:
        print("aw man. KeyError. pfft.")
        exit(-1)

    print("Setting up %s dataset." % dataset_name)
    dataset = DATASET[dataset_name](dataset_params)

    print("Creating", model_name, ". . . ")
    model_params['decode'] = False
    bot = MODELS[model_name](dataset, model_params)
    bot.train(dataset)

