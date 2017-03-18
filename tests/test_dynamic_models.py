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
        """Comparing behavior of sparse_softmax... with sampled_softmax...."""

        # 1. Here is a subset of the source code with all the irrelevant parts removed,
        #   with the aim of understanding the implementation.
        def sampled_softmax_loss(weights,
                                 biases,
                                 labels,
                                 inputs,
                                 num_sampled,
                                 num_classes):

            logits, labels =
def _compute_sampled_logits(weights,
                            biases,
                            labels,
                            inputs,
                            num_sampled,
                            num_classes,
                            num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None):


    weights = [weights]

    with tf.name_scope(name, "compute_sampled_logits", weights + [biases, inputs, labels]):
        labels = tf.cast(labels, tf.int64)
        labels_flat = tf.reshape(labels, [-1])

        # Sample the negative labels.
        #   sampled shape: [num_sampled] tensor
        #   true_expected_count shape = [batch_size, 1] tensor
        #   sampled_expected_count shape = [num_sampled] tensor
        sampled_values = tf.nn.log_uniform_candidate_sampler(
            true_classes=labels,
            num_true=num_true,
            num_sampled=num_sampled,
            unique=True,
            range_max=num_classes)

        sampled, true_expected_count, sampled_expected_count = (
            tf.stop_gradient(s) for s in sampled_values)
        sampled = tf.cast(sampled, tf.int64)

        # labels_flat is a [batch_size * num_true] tensor
        # sampled is a [num_sampled] int tensor
        all_ids = tf.concat([labels_flat, sampled], 0)

        # weights shape is [num_classes, dim]
        all_w = tf.nn.embedding_lookup(weights, all_ids, partition_strategy=partition_strategy)
        all_b = tf.nn.embedding_lookup(biases, all_ids)
        # true_w shape is [batch_size * num_true, dim]
        # true_b is a [batch_size * num_true] tensor
        true_w = tf.slice(all_w, [0, 0], tf.stack([tf.shape(labels_flat)[0], -1]))
        true_b = tf.slice(all_b, [0], tf.shape(labels_flat))

        # inputs shape is [batch_size, dim]
        # true_w shape is [batch_size * num_true, dim]
        # row_wise_dots is [batch_size, num_true, dim]
        dim = tf.shape(true_w)[1:2]
        new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
        row_wise_dots = tf.multiply(
        tf.expand_dims(inputs, 1),
        tf.reshape(true_w, new_true_w_shape))
        # We want the row-wise dot plus biases which yields a
        # [batch_size, num_true] tensor of true_logits.
        dots_as_matrix = array_ops.reshape(row_wise_dots,
        tf.concat([[-1], dim], 0))
        true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
        true_b = tf.reshape(true_b, [-1, num_true])
        true_logits += true_b

        # Lookup weights and biases for sampled labels.
        #   sampled_w shape is [num_sampled, dim]
        #   sampled_b is a [num_sampled] float tensor
        sampled_w = tf.slice(all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
        sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

        # inputs has shape [batch_size, dim]
        # sampled_w has shape [num_sampled, dim]
        # sampled_b has shape [num_sampled]
        # Apply X*W'+B, which yields [batch_size, num_sampled]
        sampled_logits = tf.matmul(
        inputs, sampled_w, transpose_b=True) + sampled_b

        if remove_accidental_hits:
        acc_hits = candidate_sampling_ops.compute_accidental_hits(
        labels, sampled, num_true=num_true)
        acc_indices, acc_ids, acc_weights = acc_hits

        # This is how SparseToDense expects the indices.
        acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
        acc_ids_2d_int32 = array_ops.reshape(
        math_ops.cast(acc_ids, dtypes.int32), [-1, 1])
        sparse_indices = array_ops.concat([acc_indices_2d, acc_ids_2d_int32], 1,
        "sparse_indices")
        # Create sampled_logits_shape = [batch_size, num_sampled]
        sampled_logits_shape = array_ops.concat(
        [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)],
        0)
        if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
        sampled_logits += sparse_ops.sparse_to_dense(
        sparse_indices,
        sampled_logits_shape,
        acc_weights,
        default_value=0.0,
        validate_indices=False)

        if subtract_log_q:
        # Subtract log of Q(l), prior probability that l appears in sampled.
        true_logits -= math_ops.log(true_expected_count)
        sampled_logits -= math_ops.log(sampled_expected_count)

        # Construct output logits and labels. The true labels/logits start at col 0.
        out_logits = array_ops.concat([true_logits, sampled_logits], 1)
        # true_logits is a float tensor, ones_like(true_logits) is a float tensor
        # of ones. We then divide by num_true to ensure the per-example labels sum
        # to 1.0, i.e. form a proper probability distribution.
        out_labels = array_ops.concat([
        array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)
        ], 1)

        return out_logits, out_labels