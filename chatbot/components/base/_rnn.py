"""Collection of base RNN classes and custom RNNCells.
"""

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from chatbot.components import bot_ops
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, LSTMStateTuple

# Required due to TensorFlow's unreliable naming across versions . . .
try:
    # r1.1
    from tensorflow.contrib.seq2seq import DynamicAttentionWrapper \
        as AttentionWrapper
    from tensorflow.contrib.seq2seq import DynamicAttentionWrapperState \
        as AttentionWrapperState
except ImportError:
    # master
    from tensorflow.contrib.seq2seq import AttentionWrapper
    from tensorflow.contrib.seq2seq import AttentionWrapperState


class Cell(RNNCell):
    """Simple wrapper class for any extensions I want to make to the
    encoder/decoder rnn cells. For now, just Dropout+GRU."""

    def __init__(self, state_size, num_layers, dropout_prob, base_cell):
        """Define the cell by composing/wrapping with tf.contrib.rnn functions.
        
        Args:
            state_size: number of units in the cell.
            num_layers: how many cells to include in the MultiRNNCell.
            dropout_prob: probability of a node being dropped.
            base_cell: (str) name of underling cell to use (e.g. 'GRUCell')
        """

        self._state_size = state_size
        self._num_layers = num_layers
        self._dropout_prob = dropout_prob
        self._base_cell = base_cell

        def single_cell():
            """Convert cell name (str) to class, and create it."""
            return getattr(tf.contrib.rnn, base_cell)(num_units=state_size)

        if num_layers == 1:
            self._cell = single_cell()
        else:
            self._cell = MultiRNNCell(
                [single_cell() for _ in range(num_layers)])

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def shape(self):
        def cell_shape():
            if "LSTM" in self._base_cell:
                return LSTMStateTuple(c=tf.TensorShape([None, self._state_size]),
                                      h=tf.TensorShape([None, self._state_size]))
            return tf.TensorShape([None, self._state_size])

        if self._num_layers == 1:
            return cell_shape()
        else:
            # tuple appears necessary for GRUCell.
            return tuple([cell_shape() for _ in range(self._num_layers)])

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`.
            state: Either 2D Tensor or tuple of 2D tensors, determined by cases:
                - `self.state_size` is int: `2-D Tensor` with shape
                    `[batch_size x self.state_size]`.
                - `self.state_size` is tuple: tuple with shapes
                    `[batch_size x s] for s in self.state_size`.
            scope: VariableScope for the created subgraph; 
                defaults to class name.

        Returns:
            A pair containing:
            - Output: 2D tensor with shape [batch_size x self.output_size].
            - New state: Either a single `2-D` tensor, or a tuple of tensors 
                matching the arity and shapes of `state`.
        """
        output, new_state = self._cell(inputs, state, scope)
        output = tf.layers.dropout(output, rate=self._dropout_prob, name="dropout")
        return output, new_state


class RNN(object):
    """Base class for encoders/decoders. Has simple instance attributes and
    an RNNCell object and getter.
    """

    def __init__(self,
                 state_size,
                 embed_size,
                 dropout_prob,
                 num_layers,
                 base_cell="GRUCell",
                 state_wrapper=None):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            embed_size: dimension size of word-embedding space.
            dropout_prob: probability of a node being dropped.
            num_layers: how many cells to include in the MultiRNNCell.
            base_cell: (str) name of underling cell to use (e.g. 'GRUCell')
            state_wrapper: allow states to store their wrapper class. See the
                wrapper method docstring below for more info.
        """
        self.state_size = state_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.base_cell = base_cell
        self._wrapper = state_wrapper

    def get_cell(self, name):
        """Returns a cell instance, defined by its name scope."""
        with tf.name_scope(name, "get_cell"):
            return Cell(state_size=self.state_size,
                        num_layers=self.num_layers,
                        dropout_prob=self.dropout_prob,
                        base_cell=self.base_cell)

    def wrapper(self, state):
        """Some RNN states are wrapped in namedtuples. 
        (TensorFlow decision, definitely not mine...). 
        
        This is here for derived classes to specify their wrapper state. 
        Some examples: LSTMStateTuple and AttentionWrapperState.
        
        Args:
            state: tensor state tuple, will be unpacked into the wrapper tuple.
        """
        if self._wrapper is None:
            return state
        else:
            return self._wrapper(*state)

    def __call__(self, *args):
        raise NotImplemented


class MyAttentionWrapper(AttentionWrapper):
    """Minor tweaks to AttentionWrapper that are useful when working with
    tf.while_loop in chat sessions.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 probability_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):

        print('outputattn:', output_attention)
        print('attnlayersize:', attention_layer_size)

        super(MyAttentionWrapper, self).__init__(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=attention_layer_size,
            alignment_history=alignment_history,
            cell_input_fn=cell_input_fn,
            probability_fn=probability_fn,
            output_attention=output_attention,
            initial_cell_state=initial_cell_state,
            name=name)

        # Assume that 'cell' is an instance of the custom 'Cell' class above.
        self._base_cell = cell._base_cell
        self._num_layers = cell._num_layers
        self._state_size = cell._state_size

    @property
    def shape(self):
        """The hoops you make me jump through, TensorFlow..."""
        if self._num_layers == 1:
            return AttentionWrapperState(
                cell_state=tf.TensorShape([None, self._state_size]),
                attention=tf.TensorShape([None, self._attention_size]),
                time=tf.TensorShape(None),
                alignment_history=())
        else:
            def layer_shape():
                return tf.TensorShape([None, self._state_size])
            return AttentionWrapperState(
                cell_state=tuple([layer_shape() for _ in range(self._num_layers)]),
                attention=tf.TensorShape([None, self._attention_size]),
                time=tf.TensorShape(None),
                alignment_history=())


class BasicRNNCell(RNNCell):
    """Same as tf.contrib.rnn.BasicRNNCell, rewritten for clarity.

    For example, many TF implementations have leftover code debt from past 
    versions, so I wanted to show what is actually going on, with the fluff 
    removed. Also, I've removed generally accepted values from parameters/args 
    in favor of just setting them.
    """

    def __init__(self, num_units, reuse=None):
        self._num_units = num_units
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Most basic RNN. Define as:
            output = new_state = act(W * input + U * state + B).
        """
        output = tf.tanh(bot_ops.linear_map(
            args=[inputs, state],
            output_size=self._num_units,
            bias=True))
        return output, output


