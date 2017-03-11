import tensorflow as tf
from utils.io_utils import EOS_ID, UNK_ID


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

    def get_embed_tensor(self, scope):
        """Returns the embedding tensor used for the given scope.
        If none exists, raises NameError.
        """
        with tf.variable_scope(scope, reuse=True):
            return tf.get_variable("embed_tensor")


class RNN(object):
    """Base class for Encoder/Decoder."""

    def __init__(self, state_size=512, embed_size=256):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            embed_size: dimension size of word-embedding space.
        """
        self.state_size = state_size
        self.embed_size = embed_size


class Encoder(RNN):
    def __init__(self, state_size=512, embed_size=256):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        super(Encoder, self).__init__(state_size=state_size, embed_size=embed_size)

    def __call__(self, inputs, return_sequence=False, initial_state=None, scope=None,):
        """Run the inputs on the encoder and return the output(s).

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            return_sequence: if True, also return the outputs at each time step.
            initial_state: (optional) Tensor with shape [batch_size, state_size] to initialize decoder cell.
            scope: (optional) variable scope name to use.

        Returns:
            outputs: (only if return_sequence is True)
                     Tensor of shape [batch_size, max_time, state_size].
            state:   The final encoder state. Tensor of shape [batch_size, state_size].
        """

        with tf.variable_scope(scope or "encoder_call"):

            #return tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            cell = tf.contrib.rnn.GRUCell(self.state_size)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs,
                                               initial_state=initial_state,
                                               dtype=tf.float32)
        if return_sequence:
            return outputs, state
        else:
            return state


class Decoder(RNN):
    """TODO
    """

    def __init__(self, state_size, output_size, embed_size, temperature=1.0):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        self.temperature = temperature
        self.output_size = output_size
        w = tf.get_variable("w", [state_size, output_size], dtype=tf.float32)
        b = tf.get_variable("b", [output_size], dtype=tf.float32)
        self.projection = (w, b)
        super(Decoder, self).__init__(state_size=state_size, embed_size=embed_size)

    def __call__(self, inputs, initial_state=None, is_chatting=False,
                 loop_embedder=None, scope=None):
        """Run the inputs on the decoder. If we are chatting, then conduct dynamic sampling,
            which is the process of generating a response given inputs == GO_ID.

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            initial_state: Tensor with shape [batch_size, state_size] to initialize decoder cell.
            is_chatting: boolean. Determines how we retrieve the outputs and the
                         returned Tensor shape.
            loop_embedder: required if is_chatting=False.
                           Embedder instance needed to feed decoder outputs as next inputs.
            scope: (optional) variable scope name to use.

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

            if not is_chatting:  # then dynamic sampling is not needed and we are done.
                return outputs, state

            if loop_embedder is None:
                raise ValueError("Loop function is required to feed decoder outputs as inputs.")

            def body(response, state):
                """Input callable for tf.while_loop. See below."""
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
                """Input callable for tf.while_loop. See below."""
                return tf.not_equal(response[-1], EOS_ID)

            # Create integer (tensor) list of output ID responses.
            response = tf.stack([self.sample(outputs)])
            # Note: This is needed so the while_loop ahead knows the shape of response.
            response = tf.reshape(response, [1,])
            tf.get_variable_scope().reuse_variables()

            # ================== BEHOLD: The tensorflow while loop. =======================
            # This allows us to sample dynamically. It also makes me happy!
            # -- Repeat 'body' while the 'cond' returns true.
            # -- 'cond' is a callable returning a boolean scalar tensor.
            # -- 'body' is a callable returning a tuple of tensors of same arity as loop_vars.
            # -- 'loop_vars' is a tuple of tensors that is passed to 'cond' and 'body'.
            response, _ = tf.while_loop(
                cond, body, (response, state),
                shape_invariants=(tf.TensorShape([None]), state.get_shape()),
                back_prop=False
            )
            # ================== FAREWELL: The tensorflow while loop. =======================

            outputs = tf.expand_dims(response, 0)
            return outputs, None

    def output_projection(self, outputs, scope=None):
        """Defines the affine transformation from state space to output space.

        Args:
            outputs: Tensor of shape [batch_size, max_time, state_size] returned by tf dynamic_rnn.
            scope: (optional) variable scope for any created here.

        Returns:
            Tensor of shape [batch_size, max_time, output_size] representing the projected outputs.
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
        """Return integer ID tensor representing the sampled word.
        """
        # Protect against extra size-1 dimensions.
        projected_output = tf.squeeze(projected_output)
        if self.temperature < 0.1:
            return tf.argmax(projected_output, axis=0)
        projected_output = tf.div(projected_output, self.temperature)
        projected_output = tf.div(tf.exp(projected_output),
                                  tf.reduce_sum(tf.exp(projected_output), axis=0))
        sample_ID = tf.squeeze(tf.multinomial(tf.expand_dims(projected_output, 0), 1))
        return sample_ID


class OutputProjection:
    """An OutputProjection applies an affine transformation to network outputs.

    Will likely be deleted soon, since functionality has now been incorporated within the
    DynamicRNN class, which was required for online chat. But hey, what if I want to project
    outwardly sometime? You can never be too sure.
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

