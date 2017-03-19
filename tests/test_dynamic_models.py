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


def _sparse_to_dense(sampled_logits, labels, sampled, num_sampled):
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

    def test_sampled_chat(self):
        """Same as test_chat but trains on new custom dynamic sampled softmax loss."""

        data_dir = '/home/brandon/terabyte/Datasets/test_data'
        dataset = TestData(data_dir)
        dataset.convert_to_tf_records('train')
        dataset.convert_to_tf_records('valid')

        print("Should I train first?")
        should_train = io_utils.get_sentence()
        is_chatting = False if should_train == 'y' else True
        print("is chatting is ", is_chatting)

        state_size = 256
        embed_size = 64
        num_layers = 3
        learning_rate = 0.1
        dropout_prob = 0.5
        ckpt_dir = 'out/sampled_st_%d_nl_%d_emb_%d_lr_%d_drop_5' % (
            state_size, num_layers, embed_size, int(100 * learning_rate)
        )

        num_samples = 40
        bot = DynamicBot(dataset,
                         num_samples=num_samples,
                         ckpt_dir=ckpt_dir,
                         batch_size=4,
                         learning_rate=learning_rate,
                         state_size=state_size,
                         embed_size=embed_size,
                         num_layers=num_layers,
                         dropout_prob=dropout_prob,
                         is_chatting=is_chatting)
        bot.compile(reset=(not is_chatting), sampled_loss=True)
        if not is_chatting:
            print("ENTERING TRAINING")
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


    def test_sampled_bot(self):
        data_dir = '/home/brandon/terabyte/Datasets/cornell'
        dataset = Cornell(data_dir, 40000)
        dataset.convert_to_tf_records('train')
        dataset.convert_to_tf_records('valid')

        is_chatting = False
        state_size = 128
        embed_size = state_size
        num_layers = 3
        learning_rate = 0.1
        dropout_prob = 0.5
        ckpt_dir = 'out'

        bot = DynamicBot(dataset,
                         ckpt_dir=ckpt_dir,
                         batch_size=32,
                         steps_per_ckpt=10,
                         learning_rate=learning_rate,
                         state_size=state_size,
                         embed_size=embed_size,
                         num_layers=num_layers,
                         dropout_prob=dropout_prob,
                         is_chatting=is_chatting)
        print('compiling')
        bot.compile(reset=(not is_chatting))
        bot.train(dataset)

    def test_sampled_softmax_from_scratch(self):
        """Comparing behavior of new dynamic_sampled_softmax_loss with a completely
        transparent version 'from scratch'. Why? Because there doesn't seem any way
        to incorporate target-weights attached to padded inputs while also using tensorflow's
        sampled_softmax_loss, as opposed to the much cleaner tf.loss.sparse_softmax_cross_entropy.

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
        def sampled_softmax_loss(output_projection, labels, state_outputs, num_sampled, vocab_size, sess=None):
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
                    true_classes=labels_flat, num_true=1, num_sampled=num_sampled,
                    unique=True, range_max=vocab_size)
                sampled, Q_true, Q_samp = (tf.stop_gradient(s) for s in sampled_values)
                sampled = tf.cast(sampled, tf.int64)

                # Casting this back to actually be flat.
                batch_times_none = tf.shape(labels_flat)[0]
                labels_flat = tf.reshape(labels, [-1])
                # Get concatenated 1D tensor of shape [batch_size * None + num_samples],
                all_ids = tf.concat([labels_flat, sampled], 0)

                # The embedding_lookup here should be thought of as embedding
                # the integer label and sampled IDs in the state space.
                # all_w has shape [batch_size * None + num_samples, state_size]
                # all_b has shape [batch_size * None + num_samples]
                all_w       = tf.nn.embedding_lookup(weights, all_ids, partition_strategy='div')
                all_b       = tf.nn.embedding_lookup(biases, all_ids)
                true_w      = tf.slice(all_w, begin=[0, 0], size=[batch_times_none, state_size])
                true_b      = tf.slice(all_b, begin=[0], size=[batch_times_none])
                sampled_w   = tf.slice(all_w, begin=[batch_times_none, 0], size=[num_sampled, state_size])
                sampled_b   = tf.slice(all_b, begin=[batch_times_none], size=[num_sampled])

                if sess is not None:
                    print('batch_times_none:', sess.run(batch_times_none))
                    print('state_size', sess.run(state_size))
                    print('shape(state_outputs)', sess.run(tf.shape(state_outputs)))
                state_outputs    = tf.reshape(state_outputs, [batch_times_none, state_size])
                state_outputs = tf.cast(state_outputs, tf.float32)
                true_logits      = tf.reduce_sum(tf.multiply(state_outputs, true_w), 1)
                true_logits     += true_b - tf.log(Q_true)
                # Matmul shapes [batch_times_none, state_size] * [state_size, num_sampled].
                sampled_logits   = tf.matmul(state_outputs, sampled_w, transpose_b=True) + sampled_b
                sampled_logits  += _sparse_to_dense(sampled_logits, tf.expand_dims(labels_flat, -1), sampled, num_sampled) - tf.log(Q_samp)
                sampled_logits  -= tf.log(Q_samp)

                # Construct output logits and labels. The true labels/logits start at col 0.
                # shape(out_logits) == [batch_times_none, 1 + num_sampled]. I'M SURE.
                out_logits = tf.concat([true_logits, sampled_logits], 1)
                # true_logits is a float tensor, ones_like(true_logits) is a float tensor of ones.
                out_labels = tf.concat([tf.ones_like(true_logits), tf.zeros_like(sampled_logits)], 1)

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=out_labels, logits=out_logits))


        # Test it works:
        seq_len = 20
        batch_size = 32
        state_size = 128
        num_sampled = 512
        vocab_size = 1024
        w = tf.get_variable("w", [state_size, vocab_size], dtype=tf.float32)
        b = tf.get_variable("b", [vocab_size], dtype=tf.float32)
        output_projection = (w, b)
        labels = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        state_outputs = np.random.random(size=(batch_size, seq_len, state_size))

        print("\nExpected quantities:")
        print("\tbatch_times_none:", batch_size * seq_len)
        print("\tstate_size:", state_size)
        print("\tshape(state_outputs):", (batch_size, seq_len, state_size))


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss = sampled_softmax_loss(output_projection, labels, state_outputs, num_sampled, vocab_size, sess=sess)
            loss = sess.run(loss)

            print('loss:\n', loss)



