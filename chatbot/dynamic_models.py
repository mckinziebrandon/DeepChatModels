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

# TODO: superclass? abc?
class DynamicBot(object):

    def __init__(self,
                 dataset,
                 batch_size=64,
                 state_size=256,
                 embed_size=64,
                 learning_rate=0.4,
                 max_seq_len=50,        # TODO: Move me.
                 num_batches=100):      # TODO: Move me.

        self.dataset     = dataset
        self.batch_size  = batch_size
        self.state_size  = state_size
        self.embed_size  = embed_size

        # TODO: don't need these stored, fix.
        self.num_batches = dataset.train_size // batch_size
        self.vocab_size  = dataset.vocab_size

        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)

        # ==============================================================================
        # Placeholders: tensors that must be fed values via feed_dict.
        # ==============================================================================

        with tf.variable_scope("placeholders"):
            # Same shape as returned by data_utils.batch_concatenate.
            # Question: we can embed both encoder and decoder inputs by feeding this
            #           same placeholder, correct? Don't see why not.
            self.batched_inputs = tf.placeholder(tf.int32, (num_batches, batch_size, max_seq_len))
            # Accepts embedded input batch array meant for encoder.
            self.encoder_inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len, embed_size])
            # Feed encoder sequence lengths for "correctness".
            self.seq_len_ph     = tf.placeholder(tf.int32, [batch_size])
            # Accepts embedded input batch array meant for decoder.
            self.decoder_inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len+1, embed_size])

        # ==============================================================================
        # Embedding operations.
        # ==============================================================================

        with tf.variable_scope("embedding_scope"):
            # Define the embedding tensor.
            embedding_params = tf.get_variable("embedding_params", [self.vocab_size, embed_size])
            # Look up all inputs at once on embedding_params. Faster than embedding wrapper.
            batch_embedded_inputs = tf.nn.embedding_lookup(embedding_params, self.batched_inputs)
            # Unpack to list of batch_sized embedded tensors.
            self.embedded_inputs = tf.unstack(batch_embedded_inputs)
            # Check type & shape results after embedding.
            for embed_sentence in self.embedded_inputs:
                if not isinstance(embed_sentence, tf.Tensor):
                    raise TypeError("Each embedded sentence should be of type Tensor.")
                if embed_sentence.shape != (batch_size, max_seq_len, embed_size):
                    raise ValueError("Embedded sentence has incorrect shape.")

        # ==============================================================================
        # DynamicRNN model.
        # ==============================================================================

        # Doesn't seem like variable scope is appropriate here, since need decoder
        # to have access to final encoder state.
        _, encoder_state = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(num_units=state_size),
                                             self.encoder_inputs,
                                             sequence_length=self.seq_len_ph,
                                             dtype=tf.float32)

        self.outputs, _ = tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(num_units=state_size),
            inputs=self.decoder_inputs,
            initial_state=encoder_state
        )

        if not isinstance(self.outputs, tf.Tensor):
            raise TypeError("Decoder outputs should be Tensor with shape"
                            "[batch_size, max_time, output_size].")

        # Plz work that would be so cool.
        target_outputs = tf.unstack(self.decoder_inputs, axis=1)[:-1]
        # Returns a scalar Tensor representing mean loss value.
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=target_outputs, logits=self.outputs
        )

        # ============================================================================
        # Training stuff.
        # ============================================================================

        params = tf.trainable_variables()
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 5.0)
        self.updates = optimizer.apply_gradients(zip(clipped_gradients, params),
                                                 global_step=self.global_step)

        # ============================================================================
        # Wrap it up. Nothing to see here.
        # ============================================================================

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def embed(self, batched_inputs):
        """Feed values in batched_inputs through model's embedder.

        Returns:
            evaluated batch-embedded input array.
        """
        input_feed = {self.batched_inputs.name: batched_inputs}
        return self.sess.run(fetches=self.embedded_inputs, feed_dict=input_feed)

    def batch_embedded_inputs(self, dataset, batch_size):
        """Embeds raw list of sentence strings and returns result.

        Args:
            sentences: list of human-readable text strings.
            max_seq_len:    size of 3rd dimension of returned array.
                        If None, defaults to length of longest sentence.

        Returns:
            numpy array of shape [num_batches, batch_size, max_seq_len],
            where num_batches == len(sentences) // batch_size.
        """

        # 1. Get list of [source_ids, target_ids] sentence id pairs.
        train_set = dataset.read_data("train")
        # 2. Split into source and target.
        source_sentences, target_sentences = np.split(train_set, 2, axis=0)
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





