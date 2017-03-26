#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
from chatbot import DynamicBot, ChatBot, SimpleBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils
from tensorflow import app

flags = app.flags
flags.DEFINE_string("config_path", "config.yml", "Location of configuration yml file.")
FLAGS = flags.FLAGS

DATASET = {'Ubuntu': Ubuntu,
           'Cornell': Cornell,
           'WMT': WMT,
           'Reddit': Reddit,
           'TestData': TestData}

MODELS = {
    'DynamicBot': DynamicBot,
    'ChatBot': ChatBot,
    'SimpleBot': SimpleBot,
}


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

