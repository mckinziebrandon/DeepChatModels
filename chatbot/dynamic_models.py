"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from utils.data_utils import GO_ID
from chatbot._train import train
from chatbot._decode import decode
from chatbot.model_components import *


def check_shape(tensor, expected_shape, log):
    if tensor.shape.as_list() != expected_shape:
        msg = "Bad shape of tensor {0}. Expected {1} but found {2}.".format(
            tensor.name, expected_shape, tensor.shape.as_list())
        log.error(msg)
        raise ValueError(msg)


class DynamicBot(object):

    def __init__(self,
                 dataset,
                 batch_size=64,
                 state_size=256,
                 embed_size=32,
                 learning_rate=0.4,
                 is_decoding=False):

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')

        self.dataset     = dataset
        self.batch_size  = batch_size
        self.state_size  = state_size
        self.embed_size  = embed_size
        self.is_decoding = is_decoding

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)

        # ==========================================================================================
        # Model Component Objects.
        # ==========================================================================================

        # Embedders.
        encoder_embedder = Embedder(dataset.vocab_size, embed_size)
        decoder_embedder = Embedder(dataset.vocab_size, embed_size)
        # DynamicRNNs.
        encoder = DynamicRNN(tf.contrib.rnn.GRUCell(state_size))
        decoder = DynamicRNN(tf.contrib.rnn.GRUCell(state_size))
        # OutputProjection.
        output_projection = OutputProjection(state_size, dataset.vocab_size)

        # ==========================================================================================
        # The sequence-to-sequence model.
        # ==========================================================================================

        # Inputs (needed by feed_dict).
        self.raw_encoder_inputs = tf.placeholder(tf.int32, (batch_size, dataset.max_seq_len))
        self.raw_decoder_inputs = tf.placeholder(tf.int32, (batch_size, dataset.max_seq_len+1))
        # Embedded input tensors.
        encoder_inputs = encoder_embedder(self.raw_encoder_inputs, scope="encoder")
        decoder_inputs = decoder_embedder(self.raw_decoder_inputs, scope="decoder")
        # Encoder-Decoder.
        encoder_state = encoder(encoder_inputs, scope="encoder")
        print(encoder_state)
        decoder_outputs, decoder_state = decoder(decoder_inputs,
                                                 scope="decoder",
                                                 initial_state=encoder_state,
                                                 return_sequence=True)
        # Projection to vocab space.
        self.outputs = output_projection(decoder_outputs)
        check_shape(self.outputs, [batch_size, dataset.max_seq_len+1, dataset.vocab_size], self.log)

        # ==========================================================================================
        # Training/evaluation operations.
        # ==========================================================================================

        # Loss - target is to predict, as output, the next decoder input.
        target_labels = self.raw_decoder_inputs[:, 1:]
        check_shape(target_labels, [batch_size, dataset.max_seq_len], self.log)
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target_labels, logits=self.outputs[:, :-1, :]
        )

        # Define gradient-descent weight-updates procedure.
        params = tf.trainable_variables()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 10.0)
        self.apply_gradients = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # Launch the session & initialize variables.
        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.global_variables())
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, encoder_inputs, decoder_inputs, forward_only=False):
        """Wrapper for self.step (below).
        """
        return self.step(encoder_inputs, decoder_inputs, forward_only)

    def step(self, encoder_inputs, decoder_inputs, forward_only=False):
        """Run forward and backward pass on single data batch.

        Args:
            encoder_inputs: shape [batch_size, max_time]
            decoder_inputs: shape [batch_size, max_time]

        Returns:
            self.is_decoding is True:
                loss: (scalar) for this batch.
            outputs: array with shape [batch_size, max_time+1, vocab_size]
        """

        decoder_inputs = [np.hstack(([GO_ID], sent)) for sent in decoder_inputs]

        input_feed = {}
        input_feed[self.raw_encoder_inputs.name] = encoder_inputs
        input_feed[self.raw_decoder_inputs.name] = decoder_inputs

        if not forward_only:
            fetches = [self.loss, self.apply_gradients]
            outputs = self.sess.run(fetches, input_feed)
            return outputs[0]  # loss
        else:
            fetches = [self.loss, self.outputs]
            outputs = self.sess.run(fetches, input_feed)
            return outputs[0], outputs[1]  # loss, outputs

    def save(self):
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join('out', "{}.ckpt".format(self.dataset.name))
        # Saves the state of all global variables.
        self.saver.save(self.sess, checkpoint_path, global_step=self.global_step)



