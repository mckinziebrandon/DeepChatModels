import tensorflow as tf
__all__ = ['Embedder', 'DynamicRNN', 'OutputProjection']


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

    def __init__(self, cell, initial_state=None):
        self.cell = cell
        self.initial_state = initial_state

    def __call__(self, inputs, scope=None,
                 return_sequence=False, initial_state=None):
        """Mimicking the tensorflow Layers API.
            Arguments:
              inputs: embedded input tensor of shape [batch_size, max_time, embed_size].
            Returns:
                outputs, state
        """

        if initial_state is not None:
            self.initial_state = initial_state
        with tf.variable_scope(scope or "dynamic_rnn_call"):
            outputs, state = tf.nn.dynamic_rnn(self.cell, inputs,
                                               initial_state=self.initial_state,
                                               dtype=tf.float32)

        if return_sequence:
            return outputs, state
        else:
            return state


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
            b, m, s = outputs.shape.as_list()
            reshaped_state = tf.reshape(outputs, [m, b, s])
            # Get projected output states; 3D Tensor.
            projected_state = tf.map_fn(single_proj, reshaped_state)
            assert(projected_state.shape == (m, b, self.output_size))
            # Return projected outputs reshaped in same general ordering as input outputs.
        return tf.reshape(projected_state, [b, m, self.output_size])



