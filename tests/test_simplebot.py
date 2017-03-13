import os
import sys
sys.path.append("..")
import tensorflow as tf
import logging
import chatbot
from utils import *
from data import Cornell

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", "out/logs", ".")
flags.DEFINE_string("dataset", "cornell", "Dataset to use. 'ubuntu', 'cornell', or 'wmt'.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, initiates chat session.")
# Integer flags.
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("state_size", 512, "Number of units in the RNN cell.")
flags.DEFINE_integer("embed_size", 64, "Size of word embedding dimension.")
flags.DEFINE_integer("nb_epoch", 10, "Number of epochs over full train set to run.")
flags.DEFINE_integer("num_layers", 3, "Num layers in underlying MultiRNNCell.")
# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
flags.DEFINE_float("temperature", 0.01, "Sampling temperature.")
flags.DEFINE_float("dropout_prob", 0.5, "Dropout rate before each layer.")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    #unittest.main()

    vocab_size = 40000
    batch_size = 32

    dataset = Cornell(vocab_size=vocab_size)

    bot = chatbot.SimpleBot(dataset.name,
                            batch_size=batch_size)

    bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)

    config = TrainConfig(FLAGS)
    print("IT'S TRAINING TIME.")
    bot.train(dataset, config)


