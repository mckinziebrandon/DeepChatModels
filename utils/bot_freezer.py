"""Utilities for freezing and unfreezing model graphs and variables on the fly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import io_utils
from pydoc import locate


def load_graph(frozen_model_path):
    """Load frozen tensorflow graph.

    Args:
        frozen_model_path: location of protobuf file containing frozen graph.

    Returns:
        tf.Graph object imported from frozen_model_path.
    """

    # Prase the frozen graph definition into a GraphDef object.
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Load the graph def into the default graph and return it.
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def freeze_bot(bot_class, freeze_path):
    """TODO: freeze_model has changed. Make sure this is up-to-date."""
    assert bot_class is not None, "Couldn't find bot class %s." % bot_class
    locate(bot_class).freeze_model(freeze_path)


def unfreeze_bot(frozen_model_path):
    """Restores the frozen graph from file and grabs input/output tensors needed to
    interface with a bot for conversation.

    Args:
        frozen_model_path: location of protobuf file containing frozen graph.

    Returns:
        outputs: tensor that can be run in a session.
    """

    bot_graph = load_graph(frozen_model_path)
    freezer = bot_graph.get_collection("freezer")
    assert isinstance(freezer, list) and len(freezer) == 2, \
    "Unknown outputs %r found in frozen graph." % freezer
    user_input, outputs = freezer
    return user_input, outputs


def unfreeze_and_chat(frozen_model_path):
    """TODO: Re-implement. This was hastily made while sketching out the model freezing
    functionality of tensorflow, as a proof of concept for website. Bad design. """

    user_input, outputs = unfreeze_bot(frozen_model_path)

    word_to_idx, _ = io_utils.get_vocab_dicts("/home/brandon/terabyte/Datasets/cornell/vocab40000.from")
    _, idx_to_word = io_utils.get_vocab_dicts("/home/brandon/terabyte/Datasets/cornell/vocab40000.to")

    encoder_inputs = io_utils.sentence_to_token_ids(
        tf.compat.as_bytes("Please don't explode."), word_to_idx)

    encoder_inputs = np.array([encoder_inputs[::-1]])
    with tf.Session() as chat_sess:
        response = chat_sess.run(outputs, feed_dict={user_input: encoder_inputs})
        response = " ".join([tf.compat.as_str(idx_to_word[i])
                             for i in response[0][:-1]])
