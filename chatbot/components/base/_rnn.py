"""Collection of base RNN classes and custom RNNCells.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMCell
from tensorflow.python.util import nest


class Cell(RNNCell):
    """Simple wrapper class for any extensions I want to make to the
    encoder/decoder rnn cells. For now, just Dropout+GRU."""

    def __init__(self, state_size, num_layers, dropout_prob, base_cell):
        """TODO
        :param state_size:
        :param num_layers:
        :param dropout_prob:
        :param base_cell:
        """
        self._state_size    = state_size
        self._num_layers    = num_layers
        self._dropout_prob = dropout_prob
        self._base_cell = base_cell

        # Convert cell name (str) to class, and create it.
        def single_cell(): return getattr(tf.contrib.rnn, base_cell)(num_units=state_size)
        if num_layers == 1:
            self._cell = single_cell()
        else:
            self._cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

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
                # Idea: return tuple([tf.TensorShape([None, self._state_size])] * 2)
                return [tf.TensorShape([None, self._state_size])] * 2
            return tf.TensorShape([None, self._state_size]) # changed from self.state_size

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
            scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
            A pair containing:
            - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
            - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`.
        """
        output, new_state = self._cell(inputs, state, scope)
        output = tf.layers.dropout(output, rate=self._dropout_prob, name="dropout")
        return output, new_state


class RNN(object):
    """Base class for BasicEncoder/DynamicDecoder."""

    def __init__(self, state_size, embed_size, dropout_prob, num_layers, base_cell="GRUCell"):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            embed_size: dimension size of word-embedding space.
        """
        self.state_size = state_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.base_cell = base_cell

    def get_cell(self, name):
        with tf.name_scope(name, "get_cell"):
            return Cell(state_size=self.state_size,
                        num_layers=self.num_layers,
                        dropout_prob=self.dropout_prob,
                        base_cell=self.base_cell)

    def __call__(self, *args):
        raise NotImplemented


class BasicRNNCell(RNNCell):
    """Same as tf.contrib.rnn.BasicRNNCell, rewritten for clarity.

    For example, many TF implementations have leftover code debt from past versions,
    so I wanted to show what is actually going on, with the fluff removed. Also, I've
    removed generally accepted values from parameters/args in favor of just setting them.
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
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        output = tf.tanh(linear_map(
            args=[inputs, state],
            output_size=self._num_units,
            bias=True))
        return output, output


def linear_map(args, output_size, biases=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        biases: tensor of shape [output_size] added to all in batch if not None.

    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    """

    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        total_arg_size += shape[1].value

    dtype = args[0].dtype

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:

        weights = tf.get_variable(
            'weights',
            [total_arg_size, output_size],
            dtype=dtype)

        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)

        return res if not biases else tf.nn.bias_add(res, biases)
