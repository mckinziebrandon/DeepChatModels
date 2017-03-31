"""Utilities for freezing and unfreezing model graphs and variables on the fly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import io_utils
import os
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
            name="freezer",
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
    inputs      = bot_graph.get_tensor_by_name("freezer/inputs")
    outputs     = bot_graph.get_tensor_by_name("freezer/outputs")
    return inputs, outputs


def unfreeze_and_chat(config):
    """TODO: Re-implement. This was hastily made while sketching out the model freezing
    functionality of tensorflow, as a proof of concept for website. Bad design. """

    # Get input and output nodes from the frozen graph def.
    # This is the bare minimum needed to chat, and thus allows for a
    # compact and efficent means of bot storage post-training.
    inputs, outputs = unfreeze_bot(config['ckpt_dir'])

    # We still need to translate the bot outputs (tokens) to english, for
    # simple humans to understand.
    from_vocab_path = os.path.join(config['data_dir'], 'vocab%d.from' % config['vocab_size'])
    to_vocab_path   = os.path.join(config['data_dir'], 'vocab%d.to' % config['vocab_size'])
    word_to_idx, _  = io_utils.get_vocab_dicts(from_vocab_path)
    _, idx_to_word  = io_utils.get_vocab_dicts(to_vocab_path)

    with tf.Session() as chat_sess:
        try:
            while True:
                # Extract sentence from human (stdin) and convert to robo-language.
                human_input     = io_utils.get_sentence()
                encoder_inputs  = io_utils.sentence_to_token_ids(
                    tf.compat.as_bytes(human_input), word_to_idx)
                encoder_inputs = np.array([encoder_inputs[::-1]])

                response = chat_sess.run(outputs, feed_dict={inputs: encoder_inputs})
                response = " ".join([tf.compat.as_str(idx_to_word[i])
                                     for i in response[0][:-1]])
                print("Robot:", response)
        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            # TODO: perhaps we also must unfreeze coordinator?

