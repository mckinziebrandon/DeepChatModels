"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import tensorflow as tf
from utils.data_utils import GO_ID
from chatbot._models import Model
from chatbot.model_components import *


def check_shape(tensor, expected_shape, log):
    if tensor.shape.as_list() != expected_shape:
        msg = "Bad shape of tensor {0}. Expected {1} but found {2}.".format(
            tensor.name, expected_shape, tensor.shape.as_list())
        log.error(msg)
        raise ValueError(msg)


class DynamicBot(Model):

    def __init__(self,
                 dataset,
                 ckpt_dir="out",
                 batch_size=64,
                 state_size=256,
                 embed_size=32,
                 learning_rate=0.4,
                 lr_decay=0.98,
                 max_seq_len=None,
                 is_decoding=False):

        if max_seq_len is None:
            max_seq_len = dataset.max_seq_len

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')
        self.dataset     = dataset
        self.state_size  = state_size
        self.embed_size  = embed_size

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
        self.raw_encoder_inputs = tf.placeholder(tf.int32, (batch_size, max_seq_len))
        self.raw_decoder_inputs = tf.placeholder(tf.int32, (batch_size, max_seq_len+1))
        # Embedded input tensors.
        encoder_inputs = encoder_embedder(self.raw_encoder_inputs, scope="encoder")
        decoder_inputs = decoder_embedder(self.raw_decoder_inputs, scope="decoder")
        # Encoder-Decoder.
        encoder_state = encoder(encoder_inputs, scope="encoder")
        decoder_outputs, decoder_state = decoder(decoder_inputs,
                                                 scope="decoder",
                                                 initial_state=encoder_state,
                                                 return_sequence=True)
        # Projection to vocab space.
        self.outputs = output_projection(decoder_outputs)
        check_shape(self.outputs, [batch_size, max_seq_len+1, dataset.vocab_size], self.log)

        # ==========================================================================================
        # Training/evaluation operations.
        # ==========================================================================================

        # Loss - target is to predict, as output, the next decoder input.
        target_labels = self.raw_decoder_inputs[:, 1:]
        check_shape(target_labels, [batch_size, max_seq_len], self.log)
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target_labels, logits=self.outputs[:, :-1, :]
        )

        # Let superclass handle the boring stuff (dirs/more instance variables).
        super(DynamicBot, self).__init__(dataset.name,
                                         ckpt_dir,
                                         dataset.vocab_size,
                                         batch_size,
                                         learning_rate,
                                         lr_decay,
                                         is_decoding)

    def compile(self, optimizer=None, max_gradient=5.0, reset=False):
        """ Configure training process and initialize model. Inspired by Keras."""

        # First, define the training portion of the graph.
        params = tf.trainable_variables()
        if optimizer is None:
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 10.0)
        self.apply_gradients = optimizer.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step)

        # Next, let superclass load param values from file (if not reset), otherwise
        # initialize newly created model.
        super(DynamicBot, self).compile(reset=reset)

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

    def train(self, encoder_inputs, decoder_inputs,
              nb_eboch=1, steps_per_ckpt=100, save_dir=None):
        i_step = 0
        avg_loss = 0.0
        avg_step_time = 0.0
        try:
            while True:
                start_time      = time.time()
                step_loss       = self.step(encoder_inputs[i_step], encoder_inputs[i_step])

                # Calculate running averages.
                avg_step_time  += (time.time() - start_time) / steps_per_ckpt
                avg_loss       += step_loss / steps_per_ckpt

                # Print updates in desired intervals (steps_per_ckpt).
                if i_step % steps_per_ckpt == 0:
                    self.save(save_dir)
                    print("Step {}: step time = {};  loss = {}".format(
                        i_step, avg_step_time, avg_loss))
                    # Reset the running averages.
                    avg_step_time = 0.0
                    avg_loss = 0.0
                i_step += 1

        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            self.save(save_dir)


    def __call__(self, encoder_inputs, decoder_inputs, forward_only=False):
        """Wrapper for self.step.
        """
        return self.step(encoder_inputs, decoder_inputs, forward_only)
