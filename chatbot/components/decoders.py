import tensorflow as tf

from tensorflow.contrib.seq2seq import DynamicAttentionWrapper
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMCell
from chatbot.components.base._rnn import RNN
from utils import io_utils
from tensorflow.python.util import nest

DYNAMIC_RNNS = {
    "dynamic_rnn": tf.nn.dynamic_rnn,
    "bidirectional_dynamic_rnn": tf.nn.bidirectional_dynamic_rnn,
    "raw_rnn": tf.nn.raw_rnn,
}


class Decoder(RNN):
    """Dynamic decoding class that supports both training and inference without
       requiring superfluous helper objects. With simple boolean parameters,
       handles the decoder sub-graph construction dynamically in its entirety.
    """

    def __init__(self,
                 base_cell,
                 state_size,
                 vocab_size,
                 embed_size,
                 dropout_prob,
                 num_layers,
                 temperature,
                 max_seq_len):
        """
        Args:
            base_cell: (str) name of RNNCell class for underlying cell.
            state_size: number of units in underlying rnn cell.
            vocab_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
        """
        super(Decoder, self).__init__(
            base_cell=base_cell,
            state_size=state_size,
            embed_size=embed_size,
            dropout_prob=dropout_prob,
            num_layers=num_layers)

        self.temperature = temperature
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        with tf.variable_scope('projection_tensors'):
            w = tf.get_variable("w", [state_size, vocab_size], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", [vocab_size], dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer())
            self._projection = (w, b)

    def __call__(self, inputs, initial_state, is_chatting, loop_embedder):
        """Run the inputs on the decoder. If we are chatting, then conduct dynamic sampling,
            which is the process of generating a response given inputs == GO_ID.

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
            initial_state: Tensor with shape [batch_size, state_size].
            is_chatting: boolean. Determines how we retrieve the outputs and the
                         returned Tensor shape.
            loop_embedder: required if is_chatting=False.
                           Embedder instance needed to feed decoder outputs as next inputs.
        Returns:
            outputs: if not chatting, tensor of shape [batch_size, max_time, vocab_size].
                     else, tensor of response IDs with shape [batch_size, max_time].
            state:   if not is_chatting, tensor of shape [batch_size, state_size].
                     else, None.
        """

        cell = self.get_cell('decoder_cell')
        self.rnn = tf.make_template(
            'decoder_rnn',
            tf.nn.dynamic_rnn,
            cell=cell,
            dtype=tf.float32
        )

        outputs, state = self.rnn(
            inputs=inputs,
            initial_state=initial_state)

        if not is_chatting:
            return outputs, state

        # Project to full output state during inference time.
        outputs = self.apply_projection(outputs)
        if loop_embedder is None:
            raise ValueError("Loop function required to feed outputs as inputs.")

        def lstm_wrapper(state):
            return LSTMStateTuple(c=state[0], h=state[1])

        def body(response, state):
            """Input callable for tf.while_loop. See below."""
            tf.get_variable_scope().reuse_variables()
            decoder_input = loop_embedder(tf.reshape(response[-1], (1, 1)),
                                          reuse=True)

            state = self._map_state_to(lstm_wrapper, state)
            if "LSTM" in self.base_cell and isinstance(state, list):
                state = tuple(state)

            outputs, state = self.rnn(
                inputs=decoder_input,
                initial_state=state,
                sequence_length=[1])

            next_id  = self.sample(self.apply_projection(outputs))
            response = tf.concat([response, tf.stack([next_id])], axis=0)
            state    = self._map_state_to(list, state)
            return response, state

        def cond(response, s):
            """Input callable for tf.while_loop. See below."""
            return tf.logical_and(tf.not_equal(response[-1], io_utils.EOS_ID),
                                  tf.less_equal(tf.size(response), self.max_seq_len))

        # Create integer (tensor) list of output ID responses.
        response = tf.stack([self.sample(outputs)])
        # Note: This is needed so the while_loop ahead knows the shape of response.
        response = tf.reshape(response, [1,], name='response')
        tf.get_variable_scope().reuse_variables()

        # ================== BEHOLD: The tensorflow while loop. ======================
        # This allows us to sample dynamically. It also makes me happy!
        # -- Repeat 'body' while the 'cond' returns true.
        # -- 'cond': callable returning a boolean scalar tensor.
        # -- 'body': callable returning a tuple of tensors of same arity as loop_vars.
        # -- 'loop_vars' is a tuple of tensors that is passed to 'cond' and 'body'.
        response, _ = tf.while_loop(
            cond, body, (response, self._map_state_to(list, state)),
            shape_invariants=(tf.TensorShape([None]), cell.shape),
            back_prop=False
        )
        # ================== FAREWELL: The tensorflow while loop. ====================

        outputs = tf.expand_dims(response, 0)
        return outputs, None

    def apply_projection(self, outputs, scope=None):
        """Defines & applies the affine transformation from state space to output space.

        Args:
            outputs: Tensor of shape [batch_size, max_time, state_size] returned by
                     tf dynamic_rnn.
            scope: (optional) variable scope for any created here.

        Returns:
            Tensor of shape [batch_size, max_time, vocab_size] representing the
            projected outputs.
        """

        with tf.variable_scope(scope, "proj_scope", [outputs]):
            # Swap 1st and 2nd indices to match expected input of map_fn.
            seq_len  = tf.shape(outputs)[1]
            st_size  = tf.shape(outputs)[2]
            time_major_outputs = tf.reshape(outputs, [seq_len, -1, st_size])
            # Project batch at single timestep from state space to output space.
            def proj_op(b): return tf.matmul(b, self._projection[0]) + self._projection[1]
            # Get projected output states; 3D Tensor with shape [batch_size, seq_len, ouput_size].
            projected_state = tf.map_fn(proj_op, time_major_outputs)
        return tf.reshape(projected_state, [-1, seq_len, self.vocab_size])

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

    def _map_state_to(self, fn, state):
        """Because LSTMStateTuple is the bane of my existence."""
        if "LSTM" not in self.base_cell: return state
        if self.num_layers > 1: return tuple(list(map(fn, state)))
        else: return fn(state)


class BasicDecoder(Decoder):

    def __call__(self, inputs, initial_state=None, is_chatting=False, loop_embedder=None):
        return super(BasicDecoder, self).__call__(
            inputs=inputs,
            initial_state=initial_state,
            is_chatting=is_chatting,
            loop_embedder=loop_embedder)


class AttentionDecoder(Decoder):
    """TODO"""

    def __init__(self, base_cell, state_size, vocab_size, embed_size,
                 dropout_prob=1.0, num_layers=2, temperature=0.0, max_seq_len=50):
        super(AttentionDecoder, self).__init__(state_size=state_size,
                                               vocab_size=vocab_size,
                                               embed_size=embed_size,
                                               dropout_prob=dropout_prob,
                                               num_layers=num_layers,
                                               temperature=temperature,
                                               max_seq_len=max_seq_len)

        self.attn = None

    def __call__(self, inputs, initial_state=None, is_chatting=False,
                 loop_embedder=None):

        self.attn = LuongAttention(
            num_units=512,
            memory=initial_state
        )
        return super(AttentionDecoder, self).__call__("bidirectional_rnn", inputs,
                                            initial_state=initial_state,
                                            is_chatting=is_chatting,
                                            loop_embedder=loop_embedder)


