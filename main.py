#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
import time
import tensorflow as tf
from chatbot import DynamicBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
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
flags.DEFINE_string("data_dir", None, "Directory containing the data files.")
flags.DEFINE_string("dataset", "cornell", "Dataset to use. 'ubuntu', 'cornell', or 'wmt'.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", False, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, initiates chat session.")
# Integer flags.
flags.DEFINE_integer("steps_per_ckpt", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("state_size", 512, "Number of units in the RNN cell.")
flags.DEFINE_integer("embed_size", None, "Models will set this to state_size if None.")
flags.DEFINE_integer("nb_epoch", 4, "Number of epochs over full train set to run.")
flags.DEFINE_integer("num_layers", 3, "Num layers in underlying MultiRNNCell.")
flags.DEFINE_integer("max_seq_len", 80, "Num layers in underlying MultiRNNCell.")
# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
flags.DEFINE_float("temperature", 0.02, "Sampling temperature.")
flags.DEFINE_float("dropout_prob", 0.5, "Dropout rate before each layer.")
FLAGS = flags.FLAGS

DATASET = {'ubuntu': Ubuntu,
           'cornell': Cornell,
           'wmt': WMT,
           'reddit': Reddit,
           'test_data': TestData}

if __name__ == "__main__":

    if FLAGS.decode:
        if FLAGS.reset_model:
            print("WARNING: To chat, should pass --reset_model=False, but found True."
                  "Resetting to False.")
            FLAGS.reset_model = False

    assert FLAGS.data_dir is not None, "You must specify --data_dir [path] as an argument."
    # If not given specific ckpt_dir, it will build a directory structure
    # rooted at out that makes for great TensorBoard visualizations.
    if FLAGS.ckpt_dir == 'out':
        FLAGS.ckpt_dir += '/' + FLAGS.dataset
        FLAGS.ckpt_dir += '/lr%d_st%d_nlay%d_drop%d' % (
            int(1e2*FLAGS.learning_rate), FLAGS.state_size,
        FLAGS.num_layers, int(1e2 * FLAGS.dropout_prob))

    # All datasets follow the same API, found in data/_dataset.py
    print("Setting up %s dataset." % FLAGS.dataset)
    dataset = DATASET[FLAGS.dataset](FLAGS.data_dir, FLAGS.vocab_size,
                                     max_seq_len=FLAGS.max_seq_len)

    # If user dataset in text format, reformat it into tensorflow protobuf
    dataset.convert_to_tf_records('train')
    dataset.convert_to_tf_records('valid')

    # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     ckpt_dir=FLAGS.ckpt_dir,
                     batch_size=FLAGS.batch_size,
                     state_size=FLAGS.state_size,
                     embed_size=FLAGS.embed_size,
                     learning_rate=FLAGS.learning_rate,
                     lr_decay=FLAGS.lr_decay,
                     steps_per_ckpt=FLAGS.steps_per_ckpt,
                     temperature=FLAGS.temperature,
                     num_layers=FLAGS.num_layers,
                     dropout_prob=FLAGS.dropout_prob,
                     is_chatting=FLAGS.decode)


    # Don't forget to compile!
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"

    if not FLAGS.decode:
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset, nb_epoch=FLAGS.nb_epoch)

    else:
        print("Initiating chat session.")
        print("Your bot has a temperature of %.2f." % FLAGS.temperature, end=" ")
        if FLAGS.temperature < 0.1:
            print("Not very adventurous, are we?")
        elif FLAGS.temperature < 0.7:
            print("This should be interesting . . . ")
        else:
            print("Enjoy your gibberish!")
            bot.decode()

