import tensorflow as tf
import pdb
from utils.io_utils import EOS_ID, UNK_ID, GO_ID, PAD_ID
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.training import bucket_by_sequence_length
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import LSTMBlockFusedCell, LSTMBlockCell, GRUBlockCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from chatbot.components._rnn import RNN, Cell


class DynamicEncoder(RNN):
    def __init__(self, state_size=512, embed_size=256, dropout_prob=1.0, num_layers=2):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        super(DynamicEncoder, self).__init__(state_size, embed_size, dropout_prob, num_layers)

    def __call__(self, inputs, return_sequence=False, scope=None, initial_state=None):
        """Run the inputs on the encoder and return the output(s).

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            return_sequence: if True, also return the outputs at each time step.
            initial_state: (optional) Tensor with shape [batch_size, state_size] to initialize decoder cell.

        Returns:
            outputs: (only if return_sequence is True)
                     Tensor of shape [batch_size, max_time, state_size].
            state:   The final encoder state. Tensor of shape [batch_size, state_size].
        """
        with tf.name_scope(scope, "encoder", values=[inputs]):

            outputs, state = tf.nn.dynamic_rnn(self.cell, inputs,
                                               initial_state=initial_state,
                                               dtype=tf.float32)

            if return_sequence:
                return outputs, state
            else:
                return state


