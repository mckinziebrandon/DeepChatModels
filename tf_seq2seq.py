"""Adapted from https://www.tensorflow.org/tutorials/seq2seq

Original header:
    Binary for training translation models and decoding from them.

    Running this program without --decode will download the WMT corpus into
    the directory specified as --data_dir and tokenize it in a very basic way,
    and then start training a model saving checkpoints to --train_dir.

    Running with --decode starts an interactive loop so you can see how
    the current checkpoint translates English sentences into French.

    See the following papers for more information on neural translation models.
     * http://arxiv.org/abs/1409.3215
     * http://arxiv.org/abs/1409.0473
     * http://arxiv.org/abs/1412.2007
"""
# Standard seq2seq model using tensorflow (version 0.12.1)
import tensorflow as tf
import numpy as np
import time
import pdb
import sys
import os

import reference.data_utils as data_utils
from reference.seq2seq_model import Seq2SeqModel

CWD      = os.getcwd()
HOME     ='/home/brandon/'
DATA_DIR = HOME + 'terabyte/Datasets/wmt'
CKPT_DIR = os.path.join(CWD, 'logs')

# Define some default param values.
QUICK_RUN       = True
MAX_STEPS       = int(30e3) if not QUICK_RUN else 1000
STEPS_PER_CKPT  = 100 if not QUICK_RUN else 50

# We use a number of buckets and pad to the closest one for efficiency.
# Each tuple corresponds with sentence length (max_source, max_target) and is for padding.
# Source (target) sentences longer than  _buckets[-1][0 (1)] will simply not be added to the dataset.

_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

# WARNING: The french data file alone has over 22 MILLION LINES. So think before changing from defaults because time exists.
# Follow-up: seriously it takes forever.
# OK WHAT: according to tutorial, this took them over 4 DAYS TO TRAIN ONE 1 EPOCH
flags = tf.app.flags
tf.app.flags.DEFINE_float("learning_rate",              0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm",          5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size",               64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size",                     512, "Size of each model layer.")           # Previous default: 1024
tf.app.flags.DEFINE_integer("num_layers",               2, "Number of layers in the model.")        # Prev default: 3
tf.app.flags.DEFINE_integer("from_vocab_size",          10000, "English vocabulary size.")          # Previous default 40000
tf.app.flags.DEFINE_integer("to_vocab_size",            10000, "French vocabulary size.")          # Previous default 40000
tf.app.flags.DEFINE_string("data_dir",                  DATA_DIR, "Data directory")
tf.app.flags.DEFINE_string("train_dir",                 CKPT_DIR, "Training (checkpoints) directory.")
tf.app.flags.DEFINE_string("from_train_data",           None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data",             None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data",             None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data",               None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size",      0, "Limit training data size (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",     STEPS_PER_CKPT, "How many training steps to do per checkpoint.") #Prev def: 200
tf.app.flags.DEFINE_boolean("decode",                   False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test",                False, "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS

def _read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def read_data(max_size=None):
    print("Preparing WMT data in %s" % FLAGS.data_dir)

    # Setup the data in appropriate directories and return desired PATHS.
    train, dev, _ = data_utils.prepare_wmt_data(FLAGS.data_dir,
                                                FLAGS.from_vocab_size,
                                                FLAGS.to_vocab_size)
    from_train, to_train = train
    from_dev, to_dev     = dev
    pdb.set_trace()

    # Read data into buckets (e.g. len(train_set) == len(buckets)).
    train_set   = _read_data(from_train, to_train, FLAGS.max_train_data_size)
    dev_set     = _read_data(from_dev, to_dev)
    return train_set, dev_set



def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    print("Calling model constructor . . .")
    model = Seq2SeqModel(FLAGS.from_vocab_size,
                         FLAGS.to_vocab_size,
                         _buckets,
                         FLAGS.size,
                         FLAGS.num_layers,
                         FLAGS.max_gradient_norm,
                         FLAGS.batch_size,
                         FLAGS.learning_rate,
                         FLAGS.learning_rate_decay_factor,
                         forward_only=forward_only,
                         dtype=tf.float32)

    # Check if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
    # If we can't, then just re-initialize model with fresh params.
    print("Checking for checkpoints . . .")
    checkpoint_state  = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if checkpoint_state and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
        print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
        model.saver.restore(session, checkpoint_state.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    """Train a en->fr translation model using WMT data."""

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        train_set, dev_set = read_data(FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size   = float(sum(train_bucket_sizes))


        print('siiicccckkk')
        exit()

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        previous_losses = []
        for i_step in range(MAX_STEPS):
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         target_weights,
                                         bucket_id,
                                         False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss      += step_loss / FLAGS.steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
                print("global step", model.global_step.eval(), end="")
                print("learning rate", model.learning_rate.eval(), end="")
                print("step time", step_time, end="")
                print("perplexity", perplexity)

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                    target_weights, bucket_id, True)
                    eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()



def main(_):
  train()

if __name__ == "__main__":
  tf.app.run()