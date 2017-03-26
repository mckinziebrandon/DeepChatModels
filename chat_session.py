#!/usr/bin/env python3
"""This shows how to run the new dynamic models (work in progress)."""
import time
import tensorflow as tf
from chatbot import DynamicBot
from data import Cornell, Ubuntu, WMT, Reddit, TestData

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

    print("using ", FLAGS.dataset)
    assert FLAGS.data_dir is not None, "You must specify --data_dir [path] as an argument."
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
                     is_chatting=True)


    print("Compiling DynamicBot.")
    bot.compile(optimizer=FLAGS.optimizer,
                max_gradient=FLAGS.max_gradient,
                sampled_loss=FLAGS.sampled_loss,
                reset=False)

    print("Initiating chat session.")
    bot.decode()

