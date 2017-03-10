import tensorflow as tf
from utils.io_utils import EOS_ID



class Embedder:
    """Acts on tensors with integer elements, embedding them in a higher-dimensional
    vector space. A single Embedder instance can embed both encoder and decoder by associating them with
    distinct scopes. """

    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def __call__(self, inputs, name=None, scope=None):
        """Mimicking the tensorflow Layers API.
            Arguments:
              inputs: input tensor of shape [batch_size, max_tgraph & summaries ime].
            Returns:
              Output tensor of shape [batch_size, max_time, embed_size]
        """
        assert len(inputs.shape) == 2, "Expected inputs rank 2 but found rank %r" % len(inputs.shape)
        with tf.variable_scope(scope or "embedding_inputs"):
            params = tf.get_variable("embed_tensor", [self.vocab_size, self.embed_size])
            embedded_inputs = tf.nn.embedding_lookup(params, inputs, name=name)
            if not isinstance(embedded_inputs, tf.Tensor):
                raise TypeError("Embedded inputs should be of type Tensor.")
            if len(embedded_inputs.shape) != 3:
                raise ValueError("Embedded sentence has incorrect shape.")
        return embedded_inputs

class RNN(object):
    """Base class for Encoder/Decoder."""
    def __init__(self, state_size=512, embed_size=256):
        self.state_size = state_size
        self.embed_size = embed_size


class Encoder(RNN):
    def __init__(self, state_size=512, embed_size=256):
        super(Encoder, self).__init__(state_size=state_size, embed_size=embed_size)

    def __call__(self, inputs, scope=None, return_sequence=False, initial_state=None):
        """Mimicking the tensorflow Layers API.
            Arguments:
              inputs: embedded input tensor of shape [batch_size, max_time, embed_size].
            Returns:
                outputs:  logits after projection.
                state:
        """

        with tf.variable_scope(scope or "encoder_call"):
            cell = tf.contrib.rnn.GRUCell(self.state_size)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                               initial_state=initial_state,
                                               dtype=tf.float32)
        if return_sequence:
            return outputs, state
        else:
            return state


class Decoder(RNN):
    def __init__(self, state_size, output_size, embed_size=256):
        self.output_size = output_size
        w = tf.get_variable("w", [state_size, output_size], dtype=tf.float32)
        b = tf.get_variable("b", [output_size], dtype=tf.float32)
        self.projection = (w, b)
        super(Decoder, self).__init__(state_size=state_size, embed_size=embed_size)

    def __call__(self, inputs, scope=None, initial_state=None,
                 is_chatting=False, loop_embedder=None):
        """


        Returns:
            outputs: if not is_chatting, tensor of shape [batch_size, max_time, output_size].
                     else, tensor of response IDs with shape [batch_size, max_time].
            state:   if not is_chatting, tensor of shape [batch_size, state_size].
                     else, None.
        """

        with tf.variable_scope(scope or "dynamic_rnn_call"):
            cell = tf.contrib.rnn.GRUCell(self.state_size)

            outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, dtype=tf.float32)
            # Outputs has shape [batch_size, max_time, output_size].
            outputs = self.output_projection(outputs)

            if not is_chatting:
                return outputs, state
            else:
                if loop_embedder is None:
                    raise ValueError("Loop function is required to feed decoder outputs as inputs.")
                # Squeeze removes all dims of dimension-1. Serves as good check here.
                # Create integer (tensor) list of output ID responses.
                response = tf.stack([self.sample(outputs)])
                # Note: This is needed so the while_loop ahead knows the shape of response.
                response = tf.reshape(response, [1,])

                tf.get_variable_scope().reuse_variables()


                def body(response, state):
                    scope.reuse_variables()
                    decoder_input = loop_embedder(tf.reshape(response[-1], (1, 1)), scope=scope)
                    outputs, state = tf.nn.dynamic_rnn(cell,
                                                 inputs=decoder_input,
                                                 initial_state=state,
                                                 sequence_length=[1],
                                                 dtype=tf.float32)
                    next_id = self.sample(self.output_projection(outputs))
                    return tf.concat([response, tf.stack([next_id])], axis=0), state

                def cond(response, s):
                    return tf.not_equal(response[-1], EOS_ID)

                response, _ = tf.while_loop(
                    cond, body, (response, state),
                    shape_invariants=(tf.TensorShape([None]), state.get_shape()),
                    back_prop=False
                )

                outputs = tf.expand_dims(response, 0)
                return outputs, None


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

        with tf.variable_scope(scope or "proj_scope"):
            # Swap 1st and 2nd indices to match expected input of map_fn.
            m  = tf.shape(outputs)[1]
            s  = tf.shape(outputs)[2]
            reshaped_state = tf.reshape(outputs, [m, -1, s])
            # Get projected output states; 3D Tensor.
            projected_state = tf.map_fn(single_proj, reshaped_state)
            # Return projected outputs reshaped in same general ordering as input outputs.
        return tf.reshape(projected_state, [-1, m, self.output_size])

    def sample(self, projected_output):
        """Return integer ID tensor representing the sampled word."""
        # Protect against extra size-1 dimensions.
        projected_output = tf.squeeze(projected_output)
        return tf.argmax(projected_output, axis=0)


class OutputProjection:
    """An OutputProjection applies an affine transformation to network outputs.
    Will likely be deleted soon, since functionality has now been incorporated within the
    DynamicRNN class, which was required for online chat.
    """

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

