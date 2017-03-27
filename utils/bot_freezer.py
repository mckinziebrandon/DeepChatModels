"""Utilities for freezing and unfreezing model graphs and variables on the fly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from pydoc import locate


def load_graph(frozen_model_path):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
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
    assert bot_class is not None, "Couldn't find bot class %s." % bot_class
    locate(bot_class).freeze_model(freeze_path)


def unfreeze_bot(frozen_model_path):
    """Burrr. Time to unfreeze the chilly bot. Ha. Heh. Woo.
    Returns:
        outputs: tensor that can be run in a session.
    """

    bot_graph = load_graph(frozen_model_path)
    outputs = bot_graph.get_collection("outputs")
    assert isinstance(outputs, list) and len(outputs) == 1, \
    "Unknown outputs %r found in frozen graph." % outputs
    outputs = outputs[0]
