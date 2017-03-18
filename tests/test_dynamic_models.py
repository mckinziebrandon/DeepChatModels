"""Run trial run on DynamicBot with the TestData Dataset."""
import os
import time
import numpy as np
import logging
import tensorflow as tf
import unittest
import sys
sys.path.append("..")
from data import *
from chatbot import DynamicBot
from utils import io_utils


def _sparse_to_dense(sampled_logits, labels, sampled):
    acc_hits = tf.nn.compute_accidental_hits(labels, sampled, num_true=1)
    acc_indices, acc_ids, acc_weights = acc_hits
    # This is how SparseToDense expects the indices.
    acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
    acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
    sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1, "sparse_indices")
    # Create sampled_logits_shape = [batch_size, num_sampled]
    sampled_logits_shape = tf.concat([tf.shape(labels)[:1], tf.expand_dims(num_sampled, 0)], 0)
    if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
    return tf.sparse_to_dense(sparse_indices, sampled_logits_shape, acc_weights,
                              default_value=0.0,validate_indices=False)


class TestDynamicModels(unittest.TestCase):


    def test_chat(self):
        """Feed the training sentences to the bot during conversation.
        It should respond somewhat predictably on these for now.
        """

        data_dir = '/home/brandon/terabyte/Datasets/test_data'
        dataset = TestData(data_dir)
        dataset.convert_to_tf_records('train')
        dataset.convert_to_tf_records('valid')

        print("Should I train first?")
        should_train = io_utils.get_sentence()
        is_chatting = False if should_train == 'y' else True
        print("is chatting is ", is_chatting)

        state_size = 2048
        embed_size = 64
        num_layers = 3
        learning_rate = 0.1
        dropout_prob = 0.5
        ckpt_dir = 'out/st_%d_nl_%d_emb_%d_lr_%d_drop_5' % (
            state_size, num_layers, embed_size, int(100 * learning_rate)
        )

        bot = DynamicBot(dataset,
                         ckpt_dir=ckpt_dir,
                         batch_size=4,
                         learning_rate=learning_rate,
                         state_size=state_size,
                         embed_size=embed_size,
                         num_layers=num_layers,
                         dropout_prob=dropout_prob,
                         is_chatting=is_chatting)
        bot.compile(reset=(not is_chatting))
        if not is_chatting:
            bot.train(dataset)
        else:
            sentence_generator = dataset.sentence_generator()
            try:
                while True:
                    sentence = next(sentence_generator)
                    print("Human:\t", sentence)
                    print("Bot:  \t", bot(sentence))
                    print()
                    time.sleep(1)
            except (KeyboardInterrupt, StopIteration):
                print('Bleep bloop. Goodbye.')



    def test_sampled_softmax(self):
        """Comparing behavior of sparse_softmax... with sampled_softmax....

        Goal: Construct a sampling loss function that can accept the following tensors:
            1. Outputs. [batch_size, None, state_size] Floats.
            2. Labels.  [batch_size, None]. Integers.
            3. Weights. [batch_size, None].

            Constraints:
                DynamicBot.compile must pass in these arguments such that
                    tf.shape(outputs[:, :, i]) == tf.shape(labels) for all/arbitrary i.
                    tf.shape(labels) == tf.shape(weights).
        """

        # 1. Here is a subset of the source code with all the irrelevant parts removed,
        # with the aim of understanding the implementation.
        # Changes:
        # - Replacing weights, biases with output_projection for clarity.
        # - Changed num_classes to vocab_size to more accurately reflect it's meaning.
        def sampled_softmax_loss(output_projection, labels, state_outputs, num_sampled, vocab_size):
            """
            Args:
                output_projection: (tuple) returned by any Decoder.get_projections_tensors()
                    - output_projection[0] == w tensor. [state_size, vocab_size]
                    - output_projection[0] == b tensor. [vocab_size]
                labels: 2D Integer tensor. [batch_size, None]
                state_outputs: 3D float Tensor [batch_size, None, state_size].
                    - In this project, usually is the decoder batch output sequence (NOT projected).
                num_sampled: number of classes out of vocab_size possible to use.
                vocab_size: total number of classes.
            """

            # Extract transpose weights, now shape is [vocab_size, state_size].
            # Use tf.reshape which is dynamic as opposed to static (i.e. slow) tf.transpose.
            weights = tf.reshape(output_projection[0], [vocab_size, -1])
            state_size = tf.shape(weights)[-1]
            biases  = output_projection[1]

            with tf.name_scope("compute_sampled_logits", [weights, biases, state_outputs, labels]):
                labels = tf.cast(labels, tf.int64)
                # Smush tensors so we can use them with tensorflow methods.
                # Question: Docs suggest we should reshape to [-1, 1] so I'm keeping.
                # but original code had it as just [-1].
                labels_flat = tf.reshape(labels, [-1, 1])
                # Sample the negative labels. Returns 3-tuple:
                #   1. sampled_candidates: [num_sampled] tensor
                #   2. true_expected_count shape = [batch_size*None, 1] tensor
                #   ---- Entries associated 1-to-1 with smushed labels.
                #   3. sampled_expected_count shape = [num_sampled] tensor
                #   ---- Entries associated 1-to-1 with sampled_candidates.
                sampled_values = tf.nn.log_uniform_candidate_sampler(
                    true_classes=labels, num_true=1, num_sampled=num_sampled,
                    unique=True, range_max=vocab_size)
                sampled, true_expected_count, sampled_expected_count = (
                    tf.stop_gradient(s) for s in sampled_values
                )
                sampled = tf.cast(sampled, tf.int64)

                # Casting this back to actually be flat.
                batch_times_none = tf.shape(labels_flat)[0]
                labels_flat = tf.reshape(labels, [-1])
                # Get concatenated 1D tensor of shape [batch_size * None + num_samples],
                all_ids = tf.concat([labels_flat, sampled], 0)

                # The embedding_lookup here should be thought of as embedding
                # the integer label and sampled IDs in the state space.
                # all_w has shape [batch_size * None + num_samples, state_size]
                all_w = tf.nn.embedding_lookup(weights, all_ids, partition_strategy='div')
                # all_b has shape [batch_size * None + num_samples]
                all_b = tf.nn.embedding_lookup(biases, all_ids)

                # Split into true and sampled projection matrices.
                true_w      = tf.slice(all_w, begin=[0, 0], size=[batch_times_none, state_size])
                true_b      = tf.slice(all_b, begin=[0], size=batch_times_none)
                sampled_w   = tf.slice(all_w, begin=[batch_times_none, 0], size=[num_sampled, state_size])
                sampled_b   = tf.slice(all_b, begin=[batch_times_none], size=[num_sampled])

                state_outputs = tf.reshape(state_outputs, [batch_times_none, state_size])
                true_logits  = tf.reduce_sum(tf.multiply(state_outputs, true_w), 1)
                true_logits += true_b

                # sampled_w has shape [num_sampled, state_size]
                # sampled_b has shape [num_sampled]
                # Apply X*W'+B, which yields [batch_size * None, num_sampled]
                sampled_logits = tf.matmul(state_outputs, sampled_w, transpose_b=True) + sampled_b

                sampled_logits += _sparse_to_dense(sampled_logits, labels, sampled)

                # Subtract log of Q(l), prior probability that l appears in sampled.
                true_logits -= tf.log(true_expected_count)
                sampled_logits -= tf.log(sampled_expected_count)

                # Construct output logits and labels. The true labels/logits start at col 0.
                out_logits = tf.concat([true_logits, sampled_logits], 1)
                # true_logits is a float tensor, ones_like(true_logits) is a float tensor
                # of ones. We then divide by num_true to ensure the per-example labels sum
                # to 1.0, i.e. form a proper probability distribution.
                out_labels = tf.concat([tf.ones_like(true_logits), tf.zeros_like(sampled_logits)], 1)

            # ============================================================================
            # Back to sampled_softmax loss function. Above is all _compute_sampled_logits.
            # ============================================================================
            logits, labels = out_logits, out_labels
            return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
