import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell


class Cell(tf.contrib.rnn.RNNCell):
    """Simple wrapper class for any extensions I want to make to the
    encoder/decoder rnn cells. For now, just Dropout+GRU."""

    def __init__(self, state_size, num_layers, dropout_prob=1.0):
        self._state_size = state_size
        self._num_layers = num_layers
        if num_layers == 1:
            self._cell = GRUCell(self._state_size)
        else:
            self._cell = MultiRNNCell([GRUCell(self._state_size) for _ in range(num_layers)])
        self._dropout_prob = dropout_prob

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def shape(self):
        if self._num_layers == 1:
            return tf.TensorShape([None, self._state_size])
        else:
            return tuple([tf.TensorShape([None, self._state_size]) for _ in range(self._num_layers)])

    def __call__(self, inputs, state, scope=None):
        inputs = tf.layers.dropout(inputs, rate=self._dropout_prob, name="dropout")
        output, new_state = self._cell(inputs, state, scope)
        return output, new_state


class RNN(object):
    """Base class for DynamicEncoder/DynamicDecoder."""

    def __init__(self, state_size, embed_size, dropout_prob, num_layers):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            embed_size: dimension size of word-embedding space.
        """
        self.state_size = state_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.cell = Cell(state_size, num_layers, dropout_prob=dropout_prob)
