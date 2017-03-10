#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
import time
import tensorflow as tf
from chatbot import DynamicBot
from data import Cornell
from utils import io_utils

# ==================================================================================================
# Parser for command-line arguments.
# - Each flag below is formatted in three columns: [name] [default value] [description]
# - Each flag's value can later be accessed via: FLAGS.name
# - The flags are shown in alphabetical order (by name).
# - Example usage:
#       python3 main.py --ckpt_dir [path_to_dir] --reset_model=False --state_size=128
# ==================================================================================================

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")

# Boolean flags.
flags.DEFINE_boolean("load_flags", False, "If true, use the same FLAGS as previous run.")
flags.DEFINE_boolean("reset_model", False, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")

# Integer flags.
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("state_size", 256, "Number of units in the RNN cell.")
flags.DEFINE_integer("embed_size", 32, "Size of word embedding dimension.")
flags.DEFINE_integer("nb_epoch", 1, "Number of epochs over full train set to run.")

# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.2, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.95, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    # This will stop the excessive INFO prints when tensorflow changes GPU pool_size.
    tf.logging.set_verbosity(tf.logging.ERROR)

    # All datasets follow the same API, found in data/_dataset.py
    dataset = Cornell(FLAGS.vocab_size)

    # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     ckpt_dir=FLAGS.ckpt_dir,
                     batch_size=FLAGS.batch_size,
                     state_size=FLAGS.state_size,
                     embed_size=FLAGS.embed_size,
                     learning_rate=FLAGS.learning_rate,
                     lr_decay=FLAGS.lr_decay,
                     is_chatting=FLAGS.decode)


    # Don't forget to compile!
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"
    if not FLAGS.decode:
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset.train_data, dataset.valid_data,
                  nb_epoch=FLAGS.nb_epoch,
                  steps_per_ckpt=FLAGS.steps_per_ckpt)

    else:
        print("Initiating chat session")
        bot.decode()




