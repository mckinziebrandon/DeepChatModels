"""Utilities for freezing and unfreezing model graphs and variables on the fly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import io_utils
import os
import re
from pydoc import locate


def load_graph(frozen_model_dir):
    """Load frozen tensorflow graph into the default graph.

    Args:
        frozen_model_dir: location of protobuf file containing frozen graph.

    Returns:
        tf.Graph object imported from frozen_model_path.
    """

    # Prase the frozen graph definition into a GraphDef object.
    frozen_file = os.path.join(frozen_model_dir, "frozen_model.pb")
    with tf.gfile.GFile(frozen_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Load the graph def into the default graph and return it.
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None
        )
    return graph


def unfreeze_bot(frozen_model_path):
    """Restores the frozen graph from file and grabs input/output tensors needed to
    interface with a bot for conversation.

    Args:
        frozen_model_path: location of protobuf file containing frozen graph.

    Returns:
        outputs: tensor that can be run in a session.
    """

    bot_graph   = load_graph(frozen_model_path)
    tensors = {'inputs': bot_graph.get_tensor_by_name('import/input_pipeline/user_input:0'),
               'outputs': bot_graph.get_tensor_by_name('import/outputs:0')}
    return tensors, bot_graph


def unfreeze_and_chat(frozen_model_path):
    """Summon a bot back from the dead and have a nice lil chat with it."""

    tensor_dict, graph = unfreeze_bot(frozen_model_path)
    config  = io_utils.parse_config(pretrained_dir=frozen_model_path)
    word_to_idx, idx_to_word = get_frozen_vocab(config)

    def as_words(sentence):
        return " ".join([tf.compat.as_str(idx_to_word[i]) for i in sentence])

    with tf.Session(graph=graph) as sess:

        def respond_to(sentence):
            """Outputs response sentence (string) given input (string)."""

            # Convert input sentence to token-ids.
            sentence_tokens = io_utils.sentence_to_token_ids(
                tf.compat.as_bytes(sentence), word_to_idx)
            sentence_tokens = np.array([sentence_tokens[::-1]])

            # Get output sentence from the chatbot.
            fetches = tensor_dict['outputs']
            feed_dict={tensor_dict['inputs']: sentence_tokens}
            response = sess.run(fetches=fetches, feed_dict=feed_dict)
            return as_words(response[0][:-1])

        sentence = io_utils.get_sentence()
        while sentence != 'exit':
            resp = respond_to(sentence)
            print("Robot:", resp)
            sentence = io_utils.get_sentence()
        print("Farewell, human.")


def get_frozen_vocab(config):
    """Helper function to get dictionaries for translating between tokens and words."""
    data_dir    = config['dataset_params']['data_dir']
    vocab_size  = config['dataset_params']['vocab_size']
    vocab_path = os.path.join(data_dir, 'vocab{}.txt'.format(vocab_size))
    word_to_idx, idx_to_word = io_utils.get_vocab_dicts(vocab_path)
    return word_to_idx, idx_to_word


class FrozenBot:

    def __init__(self, frozen_model_dir, vocab_size):
        print(frozen_model_dir)
        print(type(frozen_model_dir))
        self.tensor_dict, self.graph = unfreeze_bot(frozen_model_dir)
        self.sess = tf.Session(graph=self.graph)

        self.config = {'dataset_params': {
            'data_dir': frozen_model_dir, 'vocab_size': vocab_size}}
        self.word_to_idx, self.idx_to_word = self.get_frozen_vocab()

    def as_words(self, sentence):
        return " ".join([tf.compat.as_str(self.idx_to_word[i]) for i in sentence])

    def __call__(self, sentence):
        """Outputs response sentence (string) given input (string)."""
        # Convert input sentence to token-ids.
        sentence_tokens = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(sentence), self.word_to_idx)
        sentence_tokens = np.array([sentence_tokens[::-1]])

        # Get output sentence from the chatbot.
        fetches = self.tensor_dict['outputs']
        feed_dict={self.tensor_dict['inputs']: sentence_tokens}
        response = self.sess.run(fetches=fetches, feed_dict=feed_dict)
        return self.as_words(response[0][:-1])

