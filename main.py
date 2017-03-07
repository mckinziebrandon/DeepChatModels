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
flags.DEFINE_string("log_dir", "out/logs", "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")

# Integer flags -- First three only need custom values if you're especially worried about RAM.
flags.DEFINE_integer("max_train_samples", int(3e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("steps_per_ckpt", 50, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 20000, "English vocabulary size.")
flags.DEFINE_integer("state_size", 128, "Size of each model layer.")
flags.DEFINE_integer("embed_size", 128, "Size of word embedding dimension.")
# TODO: maybe default as None would be better here? (chooses the true dataset max sequence length.)
flags.DEFINE_integer("max_seq_len", 400, "Maximum number of words per sentence.")
flags.DEFINE_integer("nb_epoch", 1, "Number of epochs over full train set to run.")

# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.95, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

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
                     max_seq_len=FLAGS.max_seq_len,
                     is_decoding=FLAGS.decode)

    # Don't forget to compile!
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=FLAGS.max_gradient, reset=False)

    if False:
        # Get encoder/decoder training data, with shape [None, batch_size, max_seq_len].
        print("Preparing data for training.")
        encoder_sentences, decoder_sentences = io_utils.batch_concatenate(
            dataset.train_data, FLAGS.batch_size, FLAGS.max_seq_len
        )

        # Train an epoch on the data. CTRL-C at any time to safely stop training.
        # Model saved in FLAGS.ckpt_dir if specified, else "./out"
        print("Training bot. CTRL-C to stop training and start chatting.")
        bot.train(encoder_sentences, decoder_sentences,
                  nb_epoch=FLAGS.nb_epoch,
                  steps_per_ckpt=FLAGS.steps_per_ckpt)

    print("Initiating chat session")
    bot.decode()




