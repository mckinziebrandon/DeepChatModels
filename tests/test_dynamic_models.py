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
from chatbot import bot_ops
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

    def test_target_weights(self):
        """Make sure target weights set PAD targets to zero."""
        data_dir = '/home/brandon/terabyte/Datasets/test_data'
        dataset = TestData(data_dir)
        dataset.convert_to_tf_records('train')
        dataset.convert_to_tf_records('valid')

        is_chatting = False
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



        # Test it works:
        seq_len = 20
        batch_size = 64
        state_size = 128
        num_samples = 10
        vocab_size = 2000
        w = tf.get_variable("w", [state_size, vocab_size], dtype=tf.float32)
        b = tf.get_variable("b", [vocab_size], dtype=tf.float32)
        output_projection = (w, b)
        labels = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
        state_outputs = np.random.random(size=(batch_size, seq_len, state_size))
        state_outputs=tf.cast(state_outputs, tf.float32)

        print("\nExpected quantities:")
        print("\tbatch_times_none:", batch_size * seq_len)
        print("\tstate_size:", state_size)
        print("\tshape(state_outputs):", (batch_size, seq_len, state_size))


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('\n=========== FROM SCRATCH ============')
            loss = bot_ops.dynamic_sampled_softmax_loss(labels=labels,
                                                        logits=state_outputs,
                                                        output_projection=output_projection,
                                                        vocab_size=vocab_size,
                                                        from_scratch=True,
                                                        name="map_version",
                                                        num_samples=num_samples)
            loss = sess.run(loss)
            print('loss:\n', loss)

            print('\n=========== MAP VERSION ============')
            loss = bot_ops.dynamic_sampled_softmax_loss(labels=labels,
                                            logits=state_outputs,
                                            output_projection=output_projection,
                                            vocab_size=vocab_size,
                                            from_scratch=False,
                                            name="from_scratch",
                                            num_samples=num_samples)

            loss = sess.run(loss)
            print('loss:\n', loss)
            time_major_outputs = tf.reshape(state_outputs, [seq_len, batch_size, state_size])
            # Project batch at single timestep from state space to output space.
            def proj_op(bo): return tf.matmul(bo, w) + b
            # Get projected output states; 3D Tensor with shape [batch_size, seq_len, ouput_size].
            projected_state = tf.map_fn(proj_op, time_major_outputs)
            proj_out = tf.reshape(projected_state, [batch_size, seq_len, vocab_size])

            print('\n=========== ACTUAL ============')
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=proj_out)
            loss = sess.run(loss)
            print('loss:\n', loss)

