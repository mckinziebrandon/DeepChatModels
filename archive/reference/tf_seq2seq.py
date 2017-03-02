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
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf

import data_utils
from seq2seq_model import Seq2SeqModel

CWD      = os.getcwd()
HOME     ='/home/brandon/'
DATA_DIR = HOME + 'terabyte/Datasets/wmt'
CKPT_DIR = os.path.join(CWD, 'out')

DEBUG_RUN = True    # could also be called "RIDICULOUSLY_QUICK_RUN_FOR_TESTING_PURPOSES"
QUICK_RUN = False
DECODE    = False
SIZE=1024

# We use a number of buckets and pad to the closest one for efficiency.
# Source (target) sentences longer than  _buckets[-1][0 (1)] will simply not be added to the dataset.
_buckets = [(5, 10), (15, 20), (35, 45)]

if DEBUG_RUN:
    MAX_STEPS       = 500
    STEPS_PER_CKPT  = 100
    MAX_TRAIN_SAMPLES = 1e5
    SIZE=256
    _buckets = [(5, 10), (15, 20)]
elif QUICK_RUN:
    MAX_STEPS           = 2000
    STEPS_PER_CKPT      = 200
    MAX_TRAIN_SAMPLES   = int(5e6) # 0 == unlimited
else:
    MAX_STEPS = 3e5
    STEPS_PER_CKPT      = 500
    MAX_TRAIN_SAMPLES   = 0


# WARNING: The french data file alone has over 22 MILLION LINES. So think before changing from defaults because time exists.
flags = tf.app.flags
tf.app.flags.DEFINE_float("learning_rate",              0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm",          5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size",               64, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size",                     SIZE, "Size of each model layer.")           # Previous default: 1024
tf.app.flags.DEFINE_integer("num_layers",               2, "Number of layers in the model.")        # Prev default: 3
tf.app.flags.DEFINE_integer("from_vocab_size",          40000, "English vocabulary size.")          # Previous default 40000
tf.app.flags.DEFINE_integer("to_vocab_size",            40000, "French vocabulary size.")          # Previous default 40000
tf.app.flags.DEFINE_string("data_dir",                  DATA_DIR, "Data directory")
tf.app.flags.DEFINE_string("train_dir",                 CKPT_DIR, "Training (checkpoints) directory.")
tf.app.flags.DEFINE_string("from_train_data",           None, "Training data.")
tf.app.flags.DEFINE_string("to_train_data",             None, "Training data.")
tf.app.flags.DEFINE_string("from_dev_data",             None, "Training data.")
tf.app.flags.DEFINE_string("to_dev_data",               None, "Training data.")
tf.app.flags.DEFINE_integer("max_train_data_size",      MAX_TRAIN_SAMPLES, "Limit training data size (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint",     STEPS_PER_CKPT, "How many training steps to do per checkpoint.") #Prev def: 200
tf.app.flags.DEFINE_boolean("decode",                   DECODE, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test",                False, "Run a self-test if this is set to True.")
FLAGS = tf.app.flags.FLAGS

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
        train_set, dev_set = data_utils.read_data("wmt", FLAGS.data_dir,
                                                  _buckets,
                                                  FLAGS.from_vocab_size,
                                                  FLAGS.to_vocab_size,
                                                  FLAGS.max_train_data_size)
        # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        # The total number training samples, excluding the ones too long for our bucket choices.
        train_total_size   = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        # Translation: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        previous_losses = []
        for i_step in range(MAX_STEPS):
            # Sample a random bucket index according to the data distribution,
            # then get a batch of data from that bucket by calling model.get_batch.
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            # Get a batch and make a step.
            start_time = time.time()
            # Recall that target_weights are NOT parameter weights; they are weights in the sense of "weighted average."
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            print("hi")
            _, avg_perplexity, _ = model.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         target_weights,
                                         bucket_id,
                                         False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss      += avg_perplexity / FLAGS.steps_per_checkpoint

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
                print("\nglobal step:", model.global_step.eval(), end="  ")
                print("learning rate: %.4f" %  model.learning_rate.eval(), end="  ")
                print("step time: %.2f" % step_time, end="  ")
                print("perplexity: %.2f" % perplexity)

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 1 and loss > max(previous_losses[-2:]):
                    sess.run(model.lr_decay_op)
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

        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, forward_only=True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.from" % FLAGS.from_vocab_size)
        fr_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.to" % FLAGS.to_vocab_size)

        # initialize_vocabulary returns word_to_idx, idx_to_word.
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        # Decode from standard input.
        print("Type \"exit\" to exit, obviously.")
        print("Write stuff after the \">\" below and I, your robot friend, will translate it to robot French.")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            sentence = sentence[:-1]
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence longer than  truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
               outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            # Stop program if sentence == 'exit\n'.
            if sentence[:-1] == 'exit':
                print("Fine, bye :(")
                break


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
  tf.app.run()