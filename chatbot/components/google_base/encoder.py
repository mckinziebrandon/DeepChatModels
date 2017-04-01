"""
Abstract base class for encoders.
"""

from collections import namedtuple
from abc import abstractmethod
from seq2seq.configurable import Configurable
from seq2seq.graph_module import GraphModule

EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class Encoder(GraphModule, Configurable):
    """Abstract encoder class. All encoders should inherit from this.

    Args:
    params: A dictionary of hyperparameters for the encoder.
    name: A variable scope for the encoder graph.
    """

    def __init__(self, params, mode, name):
        GraphModule.__init__(self, name)
        Configurable.__init__(self, params, mode)

    def _build(self, inputs, *args, **kwargs):
        return self(inputs, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Encodes an input sequence.

        Args:
          inputs: The inputs to encode. A float32 tensor of shape [B, T, ...].
          sequence_length: The length of each input. An int32 tensor of shape [T].

        Returns:
          An `EncoderOutput` tuple containing the outputs and final state.
        """
        raise NotImplementedError
