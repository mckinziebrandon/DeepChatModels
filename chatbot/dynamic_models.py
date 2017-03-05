"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from pathlib import Path
import logging
import numpy as np
import tensorflow as tf
from utils.data_utils import batch_concatenate
from utils.data_utils import GO_ID
from chatbot._train import train
from chatbot._decode import decode
from chatbot.model_components import *

# TODO: superclass? abc?
class DynamicBot(object):

    def __init__(self,
                 dataset,
                 batch_size=64,
                 state_size=256,
                 embed_size=64,
                 max_seq_len=None,
                 learning_rate=0.4):

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')

        # TODO: make dataset compute this correctly.
        if max_seq_len is None:
            max_seq_len = dataset.max_seq_len

        self.dataset     = dataset
        self.batch_size  = batch_size
        self.state_size  = state_size
        self.embed_size  = embed_size

        # TODO: don't need these stored, fix.
        self.vocab_size  = dataset.vocab_size

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
        # Connect components from inputs to outputs to losses.
        # ==========================================================================================


        # Inputs (needed by feed_dict).
        self.raw_encoder_inputs = tf.placeholder(tf.int32, (batch_size, max_seq_len))
        self.raw_decoder_inputs = tf.placeholder(tf.int32, (batch_size, max_seq_len+1))

        # Embedded input tensors.
        encoder_inputs = encoder_embedder(self.raw_encoder_inputs, scope="encoder")
        decoder_inputs = decoder_embedder(self.raw_decoder_inputs, scope="decoder")

        # Encoder-Decoder model.
        encoder_state = encoder(encoder_inputs, scope="encoder")
        decoder_outputs, decoder_state = decoder(decoder_inputs,
                                                 scope="decoder",
                                                 initial_state=encoder_state,
                                                 return_sequence=True)

        if not isinstance(decoder_outputs, tf.Tensor):
            raise TypeError("Decoder state should be Tensor with shape"
                            "[batch_size, max_time, state_size].")

        def check_shape(tensor, expected_shape):
            if tensor.shape.as_list() != expected_shape:
                msg = "Bad shape of tensor {0}. Expected {1} but found {2}.".format(
                    tensor.name, expected_shape, tensor.shape.as_list())
                self.log.error(msg)
                raise ValueError(msg)

        # Project to vocab space (TODO: be conditional on is_decoding & do importance sampling).
        self.outputs = output_projection(decoder_outputs)
        check_shape(self.outputs, [batch_size, max_seq_len+1, self.vocab_size])

        # Target labels are just that of the next input.
        target_labels = self.raw_decoder_inputs[:, 1:]
        check_shape(target_labels, [batch_size, max_seq_len])

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target_labels, logits=self.outputs[:, :-1, :]
        )

        # ============================================================================
        # Training stuff.
        # ============================================================================

        params = tf.trainable_variables()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 10.0)
        self.updates = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # ============================================================================
        # Wrap it up. Nothing to see here.
        # ============================================================================

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, encoder_inputs, decoder_inputs):
        """
        Args:
            encoder_inputs: numpy array of shape [batch_size, max_time]
            decoder_inputs: numpy array of shape [batch_size, max_time]
        Returns:
            loss, for now
        """
        return self.step(encoder_inputs, decoder_inputs)

    def step(self, encoder_inputs, decoder_inputs):
        """Run model on single data batch.

        Args:
            encoder_inputs: shape [batch_size, max_time]
            decoder_inputs: shape [batch_size, max_time]

        Returns:
            T.B.D.
        """

        decoder_inputs = [np.hstack(([GO_ID], sent)) for sent in decoder_inputs]

        input_feed = {}
        input_feed[self.raw_encoder_inputs.name] = encoder_inputs
        input_feed[self.raw_decoder_inputs.name] = decoder_inputs

        fetches=self.loss
        return self.sess.run(fetches, input_feed)



