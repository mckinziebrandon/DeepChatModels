"""
Abstract base class for encoders.
"""

from collections import namedtuple
from abc import abstractmethod
from chatbot.components.base import GraphComponent, Component

EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class Encoder(GraphComponent, Component):
    """Abstract encoder class. All encoders should inherit from this.

    Args:
    params: A dictionary of hyperparameters for the encoder.
    name: A variable scope for the encoder graph.
    """

    def __init__(self, params, name):
        GraphComponent.__init__(self, name)
        Component.__init__(self, params)

    def _build(self, inputs, *args, **kwargs):
        return self(inputs, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError
