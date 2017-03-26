#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
import tensorflow as tf
from chatbot import DynamicBot, ChatBot, SimpleBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData
from utils import io_utils

# ==================================================================================================
# Parser for command-line arguments.
# - Each flag below is formatted in three columns: [name] [default value] [description]
# - Each flag's value can later be accessed via: FLAGS.name
# - The flags are shown in alphabetical order (by name).
# - Example usage:
#       python3 train_session.py --ckpt_dir [path_to_dir] --reset_model=False --state_size=128
# ==================================================================================================

# NOTE: Currently transitioning from FLAGS to .yml config files. Much cleaner/easier to use.

flags = tf.app.flags
# String flags -- directories and datast name(s).

flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("config_path", "config.yml", ".")
flags.DEFINE_string("data_dir", None, "Directory containing the data files.")
flags.DEFINE_string("dataset", "cornell", "Dataset to use. 'ubuntu', 'cornell', or 'wmt'.")
flags.DEFINE_string("optimizer", "Adam", "Training optimization algorithm.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", False, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, initiates chat session.")
flags.DEFINE_boolean("sampled_loss", False, "If False, uses default sparse_softmax_cross_entropy.")
# Integer flags.
flags.DEFINE_integer("steps_per_ckpt", 200, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "Number of unique words/tokens to use.")
flags.DEFINE_integer("state_size", 512, "Number of units in the RNN cell.")
flags.DEFINE_integer("embed_size", 64, "Size of embedding dimension.")
flags.DEFINE_integer("num_layers", 3, "Num layers in underlying MultiRNNCell.")
flags.DEFINE_integer("max_seq_len", 80, "Num layers in underlying MultiRNNCell.")
flags.DEFINE_integer("num_samples", 512, "subset of vocabulary_size for sampled softmax.")
# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 4.0, "Clip gradients to this value.")
flags.DEFINE_float("temperature", 0.0, "Sampling temperature.")
flags.DEFINE_float("dropout_prob", 0.2, "Dropout rate before each layer.")
flags.DEFINE_float("l1_reg", 1e-6, "l1")
FLAGS = flags.FLAGS

DATASET = {'ubuntu': Ubuntu,
           'cornell': Cornell,
           'wmt': WMT,
           'reddit': Reddit,
           'test_data': TestData}

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
    dataset = DATASET[dataset_name](data_dir=dataset_params['data_dir'],
                                    vocab_size=dataset_params['vocab_size'],
                                    max_seq_len=dataset_params['max_seq_len'])

    print("Creating", model_name, ". . . ")
    bot = MODELS[model_name](dataset, model_params)


def old_main():

    print("using ", FLAGS.dataset)
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
        # Split by directory for high-impact parameter decisions.
        FLAGS.ckpt_dir += '/optimizer_%s' % FLAGS.optimizer
        FLAGS.ckpt_dir += '/learning_rate_%.0e' % FLAGS.learning_rate
        FLAGS.ckpt_dir += '/state%d_nlay%d_l1reg%.0e_maxlen%d_drop%.0e' % (
            FLAGS.state_size, FLAGS.num_layers, FLAGS.l1_reg, FLAGS.max_seq_len, FLAGS.dropout_prob)


    # All datasets follow the same API, found in data/_dataset.py
    print("Setting up %s dataset." % FLAGS.dataset)
    dataset = DATASET[FLAGS.dataset](FLAGS.data_dir, FLAGS.vocab_size,
                                     max_seq_len=FLAGS.max_seq_len)


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
                     num_samples=FLAGS.num_samples,
                     is_chatting=FLAGS.decode)


    print("Compiling DynamicBot.")
    bot.compile(optimizer=FLAGS.optimizer,
                max_gradient=FLAGS.max_gradient,
                sampled_loss=FLAGS.sampled_loss,
                reset=FLAGS.reset_model)

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"
    if not FLAGS.decode:
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset)
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

