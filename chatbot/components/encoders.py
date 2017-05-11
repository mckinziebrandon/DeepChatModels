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

        # Concatenate each of the tuples fw and bw dimensions.
        # Now we are dealing with the concatenated "states" with dimension:
        # [batch_size, max_time, state_size * 2].
        outputs = tf.concat(outputs_tuple, -1)
        bridge = self.get_bridge("bridge", outputs.dtype)
        outputs = tf.stack([
            tf.matmul(output, bridge) for output in tf.unstack(outputs)
        ])

        # Similarly, combine the tuple of final states, resulting in:
        # [batch_size, state_size * 2].
        final_state = tf.concat(final_state_tuple, -1)

        def single_state(state):
            """Reshape bidirectional state (via fully connected layer) 
            to state size.
            """
            if 'LSTM' in self.base_cell:
                def bridge_mult(s):
                    return tf.matmul(s, bridge)
                bridged_state = LSTMStateTuple(
                    *tf.unstack(tf.map_fn(bridge_mult, state)))
            else:
                bridged_state = tf.matmul(state, bridge)
            return bridged_state

        if self.num_layers == 1:
            final_state = single_state(final_state)
        else:
            final_state = tuple([single_state(fs)
                                 for fs in tf.unstack(final_state)])

        return outputs, final_state

    def get_bridge(self, name, dtype):
        """Used for restructuring bidirectional outputs via intermediate 
        F.C. layer.
        """
        s = self.state_size
        return tf.get_variable(name, [2 * s, s], dtype=dtype)

