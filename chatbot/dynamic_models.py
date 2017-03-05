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
                 learning_rate=0.4,
                 max_seq_len=50):        # TODO: Move me.


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
        self.encoder_inputs = encoder_embedder(self.raw_encoder_inputs, scope="encoder")
        self.decoder_inputs = decoder_embedder(self.raw_decoder_inputs, scope="decoder")

        # Encoder-Decoder model.
        encoder_state = encoder(self.encoder_inputs, scope="encoder")
        decoder_outputs, decoder_state = decoder(self.decoder_inputs,
                                                 scope="decoder",
                                                 initial_state=encoder_state,
                                                 return_sequence=True)

        if not isinstance(decoder_outputs, tf.Tensor):
            raise TypeError("Decoder state should be Tensor with shape"
                            "[batch_size, max_time, state_size].")

        # Project to vocab space (TODO: be conditional on is_decoding & do importance sampling).
        projected_outputs = output_projection(decoder_outputs)

        # Question: should we feed explicit target weights?
        # Returns a scalar Tensor representing mean loss value.
        target_labels = self.raw_decoder_inputs[:, :-1]
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target_labels, logits=projected_outputs
        )

        # ============================================================================
        # Training stuff.
        # ============================================================================

        #params = tf.trainable_variables()
        #optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        #gradients = tf.gradients(self.loss, params)
        #clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        #self.updates = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                 #global_step=self.global_step)

        # ============================================================================
        # Wrap it up. Nothing to see here.
        # ============================================================================

        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())

    def embed(self, batched_inputs):
        """Feed values in batched_inputs through model's embedder.

        Returns:
            evaluated batch-embedded input array.
        """
        input_feed = {self.raw_encoder_inputs.name: batched_inputs}
        return self.sess.run(fetches=self.embedded_inputs, feed_dict=input_feed)

    def batch_embedded_inputs(self, dataset, batch_size, bettername="train"):
        """Embeds raw list of sentence strings and returns result.

        Args:
            bettername: "train", "valid", or "test". Still trying to figure out good name.
            dataset: instance of DataSet subclass containing data ifo.
            batch_size:    size of 3rd dimension of returned array.
                        If None, defaults to length of longest sentence.

        Returns:
            numpy array of shape [num_batches, batch_size, max_seq_len],
            where num_batches == len(sentences) // batch_size.
        """

        # 1. Get list of [source_ids, target_ids] sentence id pairs.
        data_ids = dataset.read_data(bettername)

        # 2. Split into source and target.
        source_sentences, target_sentences = np.split(data_ids, 2, axis=0)

        # 3. Convert both to batch format.
        batched_sources, source_lengths = batch_concatenate(
            source_sentences, batch_size, return_lengths=True)
        batched_targets, target_lengths = batch_concatenate(
            target_sentences, batch_size, return_lengths=True)

        # 4. Get embedded representation.
        embedded_sources = self.embed(batched_sources)
        embedded_targets = self.embed(batched_targets)
        return (embedded_sources, embedded_targets)


    def step(self, encoder_inputs, decoder_inputs, target_weights):
        """Run model on single data batch.

        Args:
            encoder_inputs: shape [batch_size, max_time]
            decoder_inputs: shape [batch_size, max_time]
            target_weights: TODO

        Returns:
            T.B.D.
        """

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs.name] = decoder_inputs

        # Where my gradient updates at??? [TODO]


    def __call__(self, sentence):
        """Returns output response (text string) to sentence.

        Args:
            sentence: human-readable string.

        Returns:
            response: human-redable string response to input.

        """

        # 0. Determine shape of sentences & do some formatting, considering cases:
        #       - entire dataset: split into batches.
        #       - single sentence: reset self.batch_size to 1.
        #       - handle edge cases (e.g. len(sentences) not multiple of batch_size.

        # 1. Convert sentences to list of integer sequences (data_utils.prepare_data).

        # 2. Reformat in batch-concat & padded numpy array.

        # 3. Run the session to get list of embedded inputs (not Tensors, actual float arrays).
        #       input_feed = {self.batch_concat_inputs.name: batch_sentences}
        #       self.sess.run(fetches=self.embedded_inputs, feed_dict=input_feed)

        # 4. Loop over batched embedded input arrays. Something like:
        #       for i in range(num_batches):
        #           responses.append(self.step(inputs=embeded_inputs[i],
        #                                       seq_lengths=batched_seq_len[i]))





