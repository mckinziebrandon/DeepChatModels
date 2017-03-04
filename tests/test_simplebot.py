import os
import sys
sys.path.append("..")
import tensorflow as tf
import logging
import chatbot
from utils import *

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("data_name", "ubuntu", "For now, either 'ubuntu' or 'wmt'.")
flags.DEFINE_string("data_dir", TEMP, "Directory containing the data files.")
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", "out/logs", "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
# Integer flags.
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_float("temperature", 0.0, "Sampling temperature.")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    #unittest.main()

    bot = chatbot.SimpleBot(log_dir=FLAGS.log_dir)
    FLAGS.max_train_samples = 100000
    config = TrainConfig(FLAGS)
    print("IT'S TRAINING TIME.")
    dataset = get_dataset("ubuntu")
    bot.train(dataset, config)


