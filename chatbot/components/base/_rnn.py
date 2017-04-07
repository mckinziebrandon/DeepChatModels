import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMCell

class Cell(tf.contrib.rnn.RNNCell):
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
        if num_layers == 1: self._cell = single_cell()
        else: self._cell = MultiRNNCell([single_cell() for _ in range(num_layers)])

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
                return [tf.TensorShape([None, self._state_size])] * 2
            return tf.TensorShape([None, self._state_size]) # changed from self.state_size

        if self._num_layers == 1: return cell_shape()
        else: return [cell_shape() for _ in range(self._num_layers)] # tuple may not be necessary

    def __call__(self, inputs, state, scope=None):
        """TODO
        :param inputs:
        :param state:
        :param scope:
        :return:
        """
        inputs = tf.layers.dropout(inputs, rate=self._dropout_prob, name="dropout")
        output, new_state = self._cell(inputs, state, scope)
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

