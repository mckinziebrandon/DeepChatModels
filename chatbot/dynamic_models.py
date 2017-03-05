"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard python imports.
import os
import random
from pathlib import Path
import logging

# ML/DL-specific imports.
import numpy as np
import tensorflow as tf

from utils.data_utils import batch_concatenate
# User-defined imports.
from utils import data_utils
from chatbot._train import train
from chatbot._decode import decode

# TODO: superclass? abc?
class DynamicBot(object):

    def __init__(self,
                 batch_size=64,
                 vocab_size=40000,
                 state_size=256,
                 embed_size=64,
                 max_seq_len=50,        # TODO: Move me.
                 num_batches=100):      # TODO: Move me.

        self.batch_size  = batch_size
        self.vocab_size  = vocab_size
        self.state_size  = state_size
        self.embed_size  = embed_size
        self.num_batches = num_batches

        # ==============================================================================
        # Embedding.
        # ==============================================================================

        # TODO: Move me.
        def get_embedded_inputs(batch_concat_inputs):
            with tf.variable_scope("embedded_inputs_scope"):
                embeddings = tf.get_variable("embeddings", [vocab_size, embed_size])
                batch_embedded_inputs = tf.nn.embedding_lookup(embeddings, batch_concat_inputs)
                embedded_inputs = tf.unstack(batch_embedded_inputs)
                for embed_sentence in embedded_inputs:
                    assert(isinstance(embed_sentence, tf.Tensor))
                    assert(embed_sentence.shape == (batch_size, max_seq_len, embed_size))
                return embedded_inputs

        self.batch_concat_inputs = tf.placeholder(tf.int32, (num_batches, batch_size, max_seq_len))
        self.embedded_inputs = get_embedded_inputs(self.batch_concat_inputs)

        # ==============================================================================
        # DynamicRNN model.
        # ==============================================================================

        with tf.variable_scope("model"):
            self.model_inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len, embed_size])
            self.seq_len_ph = tf.placeholder(tf.int32, [batch_size])
            cell = tf.contrib.rnn.GRUCell(num_units=state_size)
            self.outputs, state = tf.nn.dynamic_rnn(cell,
                                               self.model_inputs,
                                               sequence_length=self.seq_len_ph,
                                               dtype=tf.float32)


        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _embed_inputs(self, sentences):
        batch_sentences, sequence_lengths = batch_concatenate(
            sentences, self.batch_size, return_lengths=True
        )
        input_feed = {self.batch_concat_inputs.name: batch_sentences}
        embed_outputs = self.sess.run(fetches=self.embedded_inputs, feed_dict=input_feed)
        # TODO: Don't return sequence_lengths, no design sense.
        return embed_outputs, sequence_lengths

    def train(self, dataset):
        # TODO: Implement in all datasets.
        idx_to_word = dataset.idx_to_word()
        # TODO: Implement in all datasets.
        train_set   = dataset.read_data("train")
        sentences = [source_ids for source_ids, _ in train_set]

        embed_outputs, sequence_lengths = self._embed_inputs(sentences)
        for batch in range(self.num_batches):
            input_feed = {self.model_inputs.name: embed_outputs[batch],
                          self.seq_len_ph.name: sequence_lengths[batch]}
            pure_happiness = self.sess.run(fetches=self.outputs, feed_dict=input_feed)
            #         .-'"""""'-.
            #      .'           `.
            #     /   .      .    \
            #    :                 :
            #    |    _________    |
            #    :   \        /    :
            #     \   `.____.'    /
            #      `.           .'
            #