import tensorflow as tf
import pdb
from utils.io_utils import EOS_ID, UNK_ID, GO_ID, PAD_ID
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.training import bucket_by_sequence_length
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.rnn import LSTMBlockFusedCell, LSTMBlockCell, GRUBlockCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

__all__ = ['Encoder', 'Decoder']


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
        if self._dropout_prob > 0.1:
            inputs = tf.layers.dropout(inputs, rate=self._dropout_prob, name="dropout")
        output, new_state = self._cell(inputs, state, scope)
        return output, new_state


class RNN(object):
    """Base class for Encoder/Decoder."""

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


class Encoder(RNN):
    def __init__(self, state_size=512, embed_size=256, dropout_prob=1.0, num_layers=2,
                 scope=None):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        self.scope = scope if scope is not None else 'encoder_component'
        super(Encoder, self).__init__(state_size, embed_size, dropout_prob, num_layers)

    def __call__(self, inputs, return_sequence=False, initial_state=None):
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

        outputs, state = tf.nn.dynamic_rnn(self.cell, inputs,
                                           initial_state=initial_state,
                                           dtype=tf.float32)

        if return_sequence:
            return outputs, state
        else:
            return state


class Decoder(RNN):
    """Dynamic decoding class that supports both training and inference without
       requiring superfluous helper objects as in tensorflow's development branch.
       Based on simple boolean parameters, handles the decoder sub-graph construction
       dynamically in its entirety.
    """

    def __init__(self, state_size, output_size, embed_size,
                 dropout_prob=1.0, num_layers=2, temperature=1.0,
                 scope=None):
        """
        Args:
            state_size: number of units in underlying rnn cell.
            output_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        self.scope = scope if scope is not None else 'decoder_component'
        self.temperature = temperature
        self.output_size = output_size
        with tf.variable_scope('projection_tensors'):
            w = tf.get_variable("w", [state_size, output_size], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [output_size], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            self._projection = (w, b)
        super(Decoder, self).__init__(state_size, embed_size, dropout_prob, num_layers)

    def __call__(self, inputs, initial_state=None, is_chatting=False, loop_embedder=None):
        """Run the inputs on the decoder. If we are chatting, then conduct dynamic sampling,
            which is the process of generating a response given inputs == GO_ID.

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            initial_state: Tensor with shape [batch_size, state_size] to initialize decoder cell.
            is_chatting: boolean. Determines how we retrieve the outputs and the
                         returned Tensor shape.
            loop_embedder: required if is_chatting=False.
                           Embedder instance needed to feed decoder outputs as next inputs.
        Returns:
            outputs: if not is_chatting, tensor of shape [batch_size, max_time, output_size].
                     else, tensor of response IDs with shape [batch_size, max_time].
            state:   if not is_chatting, tensor of shape [batch_size, state_size].
                     else, None.
        """

        with tf.variable_scope("decoder_call") as dec_scope:
            outputs, state = tf.nn.dynamic_rnn(
                self.cell, inputs, initial_state=initial_state, dtype=tf.float32
            )

            if not is_chatting:
                return outputs, state

            # Project to full output state during inference time.
            outputs = self.apply_projection(outputs)
            if loop_embedder is None:
                raise ValueError("Loop function is required to feed decoder outputs as inputs.")

            def body(response, state):
                """Input callable for tf.while_loop. See below."""
                dec_scope.reuse_variables()
                with tf.variable_scope(self.scope):
                    decoder_input, _ = loop_embedder(tf.reshape(response[-1], (1, 1)), reuse=True)
                outputs, state = tf.nn.dynamic_rnn(self.cell,
                                             inputs=decoder_input,
                                             initial_state=state,
                                             sequence_length=[1],
                                             dtype=tf.float32)
                next_id = self.sample(self.apply_projection(outputs))
                return tf.concat([response, tf.stack([next_id])], axis=0), state

            def cond(response, s):
                """Input callable for tf.while_loop. See below."""
                return tf.logical_or(tf.not_equal(response[-1], EOS_ID),
                                     tf.not_equal(response[-1], PAD_ID))

            # Create integer (tensor) list of output ID responses.
            response = tf.stack([self.sample(outputs)])
            # Note: This is needed so the while_loop ahead knows the shape of response.
            response = tf.reshape(response, [1,], name='response')
            dec_scope.reuse_variables()

            # ================== BEHOLD: The tensorflow while loop. =======================
            # This allows us to sample dynamically. It also makes me happy!
            # -- Repeat 'body' while the 'cond' returns true.
            # -- 'cond' is a callable returning a boolean scalar tensor.
            # -- 'body' is a callable returning a tuple of tensors of same arity as loop_vars.
            # -- 'loop_vars' is a tuple of tensors that is passed to 'cond' and 'body'.
            response, _ = tf.while_loop(
                cond, body, (response, state),
                shape_invariants=(tf.TensorShape([None]), self.cell.shape),
                back_prop=False
            )
            # ================== FAREWELL: The tensorflow while loop. =======================

            outputs = tf.expand_dims(response, 0)
            return outputs, None

    def apply_projection(self, outputs, scope=None):
        """Defines & applies the affine transformation from state space to output space.

        Args:
            outputs: Tensor of shape [batch_size, max_time, state_size] returned by tf dynamic_rnn.
            scope: (optional) variable scope for any created here.

        Returns:
            Tensor of shape [batch_size, max_time, output_size] representing the projected outputs.
        """

        with tf.name_scope(scope, "proj_scope", [outputs]):
            # Swap 1st and 2nd indices to match expected input of map_fn.
            seq_len  = tf.shape(outputs)[1]
            st_size  = tf.shape(outputs)[2]
            time_major_outputs = tf.reshape(outputs, [seq_len, -1, st_size])
            # Project batch at single timestep from state space to output space.
            def proj_op(b): return tf.matmul(b, self._projection[0]) + self._projection[1]
            # Get projected output states; 3D Tensor with shape [batch_size, seq_len, ouput_size].
            projected_state = tf.map_fn(proj_op, time_major_outputs)
        return tf.reshape(projected_state, [-1, seq_len, self.output_size])

    def sample(self, projected_output):
        """Return integer ID tensor representing the sampled word.
        """
        with tf.name_scope('decoder_sampler', values=[projected_output]):
            # Protect against extra size-1 dimensions.
            projected_output = tf.squeeze(projected_output)
            if self.temperature < 0.02:
                return tf.argmax(projected_output, axis=0)
            projected_output = tf.div(projected_output, self.temperature)
            projected_output = tf.div(tf.exp(projected_output),
                                      tf.reduce_sum(tf.exp(projected_output), axis=0))
            sample_ID = tf.squeeze(tf.multinomial(tf.expand_dims(projected_output, 0), 1))
        return sample_ID

    def get_projection_tensors(self):
        """Returns the tuple (w, b) that decoder uses for projecting.
        Required as argument to the sampled softmax loss.
        """
        return self._projection
