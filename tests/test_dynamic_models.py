"""Complete mock model creation, training, and decoding from start to finish."""

import numpy as np
import logging
import sys
sys.path.append("..")

import tensorflow as tf
from utils.test_data import TestData
from chatbot.dynamic_models import DynamicBot


if __name__ == '__main__':

    # Get dataset and its properties.
    dataset = TestData()

    bot = DynamicBot(dataset)

