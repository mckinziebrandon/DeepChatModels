"""Train seq2seq attention model on ubuntu dialogue corpus."""
import os
import sys
import time

import tensorflow as tf
import utils.data_utils
from chatbot.model import Chatbot
from utils import *



def train(chatbot):
    """ Train chatbot using 1-on-1 ubuntu dialogue corpus. """

    with tf.Session() as sess:
        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % MAX_TRAIN_SAMPLES)
        train_set, dev_set = data_utils.read_data("ubuntu",
                                                  DATA_DIR,
                                                  _buckets,
                                                  VOCAB_SIZE,
                                                  max_train_data_size=MAX_TRAIN_SAMPLES)

        # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        # The total number training samples, excluding the ones too long for our bucket choices.
        train_total_size   = float(sum(train_bucket_sizes))

        # TODO: dont 4get this is here.
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

            _, avg_perplexity, _ = model.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         target_weights,
                                         bucket_id,
                                         False)

            step_time += (time.time() - start_time) / STEPS_PER_CKPT
            loss      += avg_perplexity / STEPS_PER_CKPT

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % STEPS_PER_CKPT == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
                print("\nglobal step:", model.global_step.eval(), end="  ")
                print("learning rate: %.4f" %  model.learning_rate.eval(), end="  ")
                print("step time: %.2f" % step_time, end="  ")
                print("perplexity: %.2f" % perplexity)

                # Decrease learning rate more aggressively.
                if len(previous_losses) > 3 and loss > min(previous_losses[-3:]):
                    sess.run(model.lr_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(CKPT_DIR, "converse.ckpt")
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

        checkpoint_path = os.path.join(CKPT_DIR, "converse.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

