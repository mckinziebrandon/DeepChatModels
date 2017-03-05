"""Complete mock model creation, training, and decoding from start to finish."""

import numpy as np
import logging
import sys
sys.path.append("..")
import tensorflow as tf
from utils.data_utils import batch_concatenate
from utils import TestData


def get_embedded_inputs(batch_concat_inputs):

    with tf.variable_scope("embedded_inputs_scope"):
        # 1. Embed the inputs.
        embeddings = tf.get_variable("embeddings", [vocab_size, embed_size])
        batch_embedded_inputs = tf.nn.embedding_lookup(embeddings, batch_concat_inputs)
        # NO WAY. THIS IS AWESOME!!!
        embedded_inputs = tf.unstack(batch_embedded_inputs)
        for embed_sentence in embedded_inputs:
            assert(isinstance(embed_sentence, tf.Tensor))
            assert(embed_sentence.shape == (batch_size, max_seq_len, embed_size))
        return embedded_inputs


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('MainLogger')

    batch_size  = 2
    vocab_size  = 100
    state_size  = 128
    embed_size  = 64

    # Get the data in format of word ID lists.
    dataset = TestData(vocab_size)
    idx_to_word = dataset.idx_to_word()
    train_set = dataset.read_data("train")

    # ==============================================================================
    # Manual data preprocessing.
    # ==============================================================================

    # Just get a list of sentence ids for this simple test.
    sentences = [source_ids for source_ids, _ in train_set]
    batch_sentences, sequence_lengths = batch_concatenate(
        sentences, batch_size, return_lengths=True
    )
    max_seq_len = sequence_lengths.max()

    # ==============================================================================
    # Embedding.
    # ==============================================================================

    batch_concat_inputs = tf.placeholder(tf.int32, batch_sentences.shape)
    embedded_inputs = get_embedded_inputs(batch_concat_inputs)

    # ==============================================================================
    # DynamicRNN model.
    # ==============================================================================

    with tf.variable_scope("model"):
        model_inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len, embed_size])
        seq_len_ph = tf.placeholder(tf.int32, [batch_size])
        cell = tf.contrib.rnn.GRUCell(num_units=state_size)
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           model_inputs,
                                           sequence_length=seq_len_ph,
                                           dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    # ==============================================================================
    # Execution.
    # ==============================================================================

    with tf.Session() as sess:

        sess.run(init_op)

        input_feed = {batch_concat_inputs.name: batch_sentences}
        embed_outputs = sess.run(fetches=embedded_inputs, feed_dict=input_feed)

        num_batches = batch_sentences.shape[0]
        for batch in range(num_batches):
            input_feed = {model_inputs.name: embed_outputs[batch],
                          seq_len_ph.name: sequence_lengths[batch]}
            pure_happiness = sess.run(fetches=outputs, feed_dict=input_feed)
            #         .-'"""""'-.
            #      .'           `.
            #     /   .      .    \
            #    :                 :
            #    |    _________    |
            #    :   \        /    :
            #     \   `.____.'    /
            #      `.           .'
            #        `-._____.-'
