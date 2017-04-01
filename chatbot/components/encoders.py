import copy
import tensorflow as tf
from chatbot.components._rnn import RNN, Cell
from chatbot.components.google_base import Encoder, EncoderOutput
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell


def _default_encoder_params():
    return {
        "cell_class": "GRUCell",
        "cell_params": {
          "num_units": 512
        },
        "dropout_input_keep_prob": 1.0,
        "dropout_output_keep_prob": 1.0,
        "num_layers": 1,
        "state_size": 512,
        "embed_size": 64,
        "dropout_prob": 0.2,
        "num_layers": 3
    }


class UniEncoder(Encoder):
    #def __init__(self, params, mode, name="forward_rnn_encoder"):
    def __init__(self, state_size=512, embed_size=256, dropout_prob=1.0, num_layers=2):
        params = self.default_params()
        params['rnn_cell']['state_size'] = state_size
        params['rnn_cell']['embed_size'] = embed_size
        params['rnn_cell']['dropout_prob'] = dropout_prob
        params['rnn_cell']['num_layers'] = num_layers
        super(UniEncoder, self).__init__(params, "uniencoder")

    @staticmethod
    def default_params():
        return {"rnn_cell": _default_encoder_params()}

    def __call__(self, inputs, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(-0.04, 0.04))
        cell = GRUCell(512)
        _, state = tf.nn.dynamic_rnn(cell,
                                     inputs,
                                     dtype=tf.float32)
        return None, state


class BasicEncoder(RNN):
    def __init__(self, state_size=512, embed_size=256, dropout_prob=1.0, num_layers=2):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        super(BasicEncoder, self).__init__(state_size, embed_size, dropout_prob, num_layers)

    def __call__(self, inputs, initial_state=None, scope=None):
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

            cell = self.get_cell("basic_enc_cell")
            _, state = tf.nn.dynamic_rnn(cell,
                                         inputs,
                                         initial_state=initial_state,
                                         dtype=tf.float32)
            return None, state


class BidirectionalEncoder(RNN):
    def __init__(self, state_size=512, embed_size=256, dropout_prob=1.0, num_layers=2):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        super(BidirectionalEncoder, self).__init__(state_size, embed_size, dropout_prob, num_layers)

    def __call__(self, inputs, initial_state=None, scope=None):
        """Run the inputs on the encoder and return the output(s).

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].

        Returns:
            outputs: Tensor of shape [batch_size, max_time, state_size].
            state:   The final encoder state. Tensor of shape [batch_size, state_size].
        """
        with tf.name_scope(scope, "encoder", values=[inputs]):

            cell_fw = self.get_cell("cell_fw")
            cell_bw = self.get_cell("cell_bw")
            outputs_tuple, _ =  tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=inputs,
                dtype=tf.float32)

            # This is not the best way to convert shapes, but it works.
            # TODO: improve this please.
            outputs = tf.concat(outputs_tuple, 2)
            final_state = outputs[:, -1, :]  # [batch_size, 2*state_size]
            bridge = tf.get_variable("bridge",
                                     [2*self.state_size, self.state_size],
                                     dtype=final_state.dtype)
            final_state = tf.matmul(final_state,
                                    bridge,
                                    name="final_state_matmul")
            return outputs, final_state



