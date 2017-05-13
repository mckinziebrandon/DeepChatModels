"""Classes for the dynamic encoders."""

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMCell
from chatbot.components.base._rnn import RNN
from tensorflow.python.layers import core as layers_core


class BasicEncoder(RNN):
    """Encoder architecture that is defined by its cell running 
    inside dynamic_rnn.
    """

    def __call__(self, inputs, initial_state=None):
        """Run the inputs on the encoder and return the output(s).

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            initial_state: (optional) Tensor with shape [batch_size, state_size] 
                to initialize decoder cell.

        Returns:
            outputs: (only if return_sequence is True)
                     Tensor of shape [batch_size, max_time, state_size].
            state:   The final encoder state; shape [batch_size, state_size].
        """

        cell = self.get_cell("basic_enc_cell")
        _, state = tf.nn.dynamic_rnn(cell,
                                     inputs,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
        return _, state


class BidirectionalEncoder(RNN):
    """Encoder that concatenates two copies of its cell forward and backward and
    feeds into a bidirectional_dynamic_rnn.

    Outputs are concatenated before being returned. I may move this 
    functionality to an intermediate class layer that handles shape-matching 
    between encoder/decoder.
    """

    def __call__(self, inputs, initial_state=None):
        """Run the inputs on the encoder and return the output(s).

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].

        Returns:
            outputs: Tensor of shape [batch_size, max_time, state_size].
            state: The final encoder state; shape [batch_size, state_size].
        """

        cell_fw = self.get_cell("cell_fw")
        cell_bw = self.get_cell("cell_bw")
        outputs_tuple, final_state_tuple = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            dtype=tf.float32)

        # Create fully connected layer to help get us back to
        # state size (from the dual state fw-bw).
        layer = layers_core.Dense(units=self.state_size, use_bias=False)

        def single_state(state):
            """Reshape bidirectional state (via fully connected layer)
            to state size.
            """
            if 'LSTM' in self.base_cell:
                bridged_state = LSTMStateTuple(
                    c=layer(state[0]),
                    h=layer(state[1]))
            else:
                bridged_state = layer(state)
            return bridged_state

        # Concatenate each of the tuples fw and bw dimensions.
        # Now we are dealing with the concatenated "states" with dimension:
        # [batch_size, max_time, state_size * 2].
        # NOTE: Convention of LSTMCell is that outputs only contain the
        # the hidden state (i.e. 'h' only, no 'c').
        outputs = tf.concat(outputs_tuple, -1)
        outputs = tf.map_fn(layer, outputs)

        # Similarly, combine the tuple of final states, resulting in:
        # [batch_size, state_size * 2].
        final_state = tf.concat(final_state_tuple, -1)

        if self.num_layers == 1:
            final_state = single_state(final_state)
        else:
            final_state = tuple([single_state(fs)
                                 for fs in tf.unstack(final_state)])

        return outputs, final_state

