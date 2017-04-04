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
    tensors = {
        'inputs': bot_graph.get_tensor_by_name('import/input_pipeline/input_pipeline/user_input_ph:0'),
        'outputs': bot_graph.get_tensor_by_name('import/decoder/decoder_1/ExpandDims:0')
    }
    return tensors, bot_graph


def unfreeze_and_chat(frozen_model_path):
    """Summon a bot back from the dead and have a nice lil chat with it."""

    tensor_dict, graph = unfreeze_bot(frozen_model_path)
    config  = io_utils.parse_config(frozen_model_path)
    dataset = locate(config['dataset'])(config['dataset_params'])

    with tf.Session(graph=graph) as sess:

        def respond_to(sentence):
            """Outputs response sentence (string) given input (string)."""
            # Convert input sentence to token-ids.
            sentence_tokens = io_utils.sentence_to_token_ids(
                tf.compat.as_bytes(sentence), dataset.word_to_idx)
            sentence_tokens = np.array([sentence_tokens[::-1]])
            # Get output sentence from the chatbot.
            response = sess.run(tensor_dict['outputs'],
                                feed_dict={tensor_dict['inputs']: sentence_tokens})
            return dataset.as_words(response[0][:-1])

        sentence = io_utils.get_sentence()
        while sentence != 'exit':
            resp = respond_to(sentence)
            print("Robot:", resp)
            sentence = io_utils.get_sentence()
        print("Farewell, human.")
