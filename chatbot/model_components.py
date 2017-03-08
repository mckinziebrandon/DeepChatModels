import tensorflow as tf
from utils.io_utils import EOS_ID



class Embedder:
    """Acts on tensors with integer elements, embedding them in a higher-dimensional
    vector space."""

    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def __call__(self, inputs, scope=None):
        """Mimicking the tensorflow Layers API.
            Arguments:
              inputs: input tensor of shape [batch_size, max_time].
            Returns:
              Output tensor of shape [batch_size, max_time, embed_size]
        """
        assert(len(inputs.shape) == 2)
        with tf.variable_scope(scope or "embedding_inputs"):
            params = tf.get_variable("params", [self.vocab_size, self.embed_size])
            embedded_inputs = tf.nn.embedding_lookup(params, inputs)
            if not isinstance(embedded_inputs, tf.Tensor):
                raise TypeError("Embedded inputs should be of type Tensor.")
            if len(embedded_inputs.shape) != 3:
                raise ValueError("Embedded sentence has incorrect shape.")
        return embedded_inputs


class DynamicRNN:
    """Wrapper class for tensorflow's dynamic_rnn, since I prefer OOP."""

    def __init__(self, state_size, output_size, embed_size=128, initial_state=None):
        self.state_size = state_size
        self.initial_state = initial_state
        self.state_size = state_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.out_to_embed = tf.get_variable("out_to_embed", [self.output_size, embed_size])
        w = tf.get_variable("w", [state_size, output_size], dtype=tf.float32)
        b = tf.get_variable("b", [output_size], dtype=tf.float32)
        self.projection = (w, b)

    def __call__(self, inputs, scope=None,
                 return_sequence=False, initial_state=None, is_decoding=False):
        """Mimicking the tensorflow Layers API.
            Arguments:
              inputs: embedded input tensor of shape [batch_size, max_time, embed_size].
            Returns:
                outputs:  logits after projection.
                state:
        """

        if initial_state is not None:
            self.initial_state = initial_state
        with tf.variable_scope(scope or "dynamic_rnn_call"):
            cell = tf.contrib.rnn.GRUCell(self.state_size)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                               initial_state=self.initial_state,
                                               dtype=tf.float32)
            # Outputs has shape [batch_size, max_time, output_size].
            outputs = self.output_projection(outputs)

            # outputs.shape in this case is [1, 1, output_size].
            output_logits = [outputs[0][0]]
            pred = tf.argmax(output_logits[-1])
            if is_decoding:
                tf.get_variable_scope().reuse_variables()
                output_length = 1
                while output_length <= 20:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = tf.nn.embedding_lookup(self.out_to_embed, tf.argmax(output_logits[-1]))
                        inp = tf.reshape(inp, [-1, -1, self.embed_size])
                    outputs, state = tf.nn.dynamic_rnn(cell, inp, initial_state=state, dtype=tf.float32)
                    outputs = self.output_projection(outputs)
                    output_logits.append(outputs[0][0])
                    output_length += 1
                outputs = tf.reshape(tf.stack(output_logits), [-1, -1, self.output_size])

        if return_sequence:
            return outputs, state
        else:
            return state

    def output_projection(self, outputs, scope=None):
        """
        :param outputs: [batch_size, max_time, state_size] (1st output from dynamic_rnn)
        :return: projected outputs with shape [batch_size, max_time, output_size]
        """

        def single_proj(single_batch):
            """
            :param single_batch: [batch_size, state_size]
            :return: tensor shape [batch_size, output_size]
            """
            return tf.matmul(single_batch, self.projection[0]) + self.projection[1]

        with tf.variable_scope(scope or "output_projection_call"):
            # Swap 1st and 2nd indices to match expected input of map_fn.
            m  = tf.shape(outputs)[1]
            s  = tf.shape(outputs)[2]
            reshaped_state = tf.reshape(outputs, [m, -1, s])
            # Get projected output states; 3D Tensor.
            projected_state = tf.map_fn(single_proj, reshaped_state)
            # Return projected outputs reshaped in same general ordering as input outputs.
        return tf.reshape(projected_state, [-1, m, self.output_size])


class OutputProjection:
    """An OutputProjection applies an affine transformation to network outputs."""

    def __init__(self, state_size, output_size):
        self.state_size = state_size
        self.output_size = output_size
        w = tf.get_variable("w", [state_size, output_size], dtype=tf.float32)
        b = tf.get_variable("b", [output_size], dtype=tf.float32)
        self.projection = (w, b)

    def __call__(self, outputs, scope=None):
        """
        :param outputs: [batch_size, max_time, state_size] (1st output from dynamic_rnn)
        :return: projected outputs with shape [batch_size, max_time, output_size]
        """

        def single_proj(single_batch):
            """
            :param single_batch: [batch_size, state_size]
            :return: tensor shape [batch_size, output_size]
            """
            return tf.matmul(single_batch, self.projection[0]) + self.projection[1]

        with tf.variable_scope(scope or "output_projection_call"):
            # Swap 1st and 2nd indices to match expected input of map_fn.
            #_, m, s = outputs.shape.as_list()
            m  = tf.shape(outputs)[1]
            s  = tf.shape(outputs)[2]
            reshaped_state = tf.reshape(outputs, [m, -1, s])
            # Get projected output states; 3D Tensor.
            projected_state = tf.map_fn(single_proj, reshaped_state)
            # Return projected outputs reshaped in same general ordering as input outputs.
        return tf.reshape(projected_state, [-1, m, self.output_size])

