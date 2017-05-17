import logging
import tensorflow as tf
import sys

# Required due to TensorFlow's unreliable naming across versions . . .
try:
    # r1.1
    from tensorflow.contrib.seq2seq import DynamicAttentionWrapperState \
        as AttentionWrapperState
except ImportError:
    # master
    from tensorflow.contrib.seq2seq import AttentionWrapperState

from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention
from tensorflow.contrib.rnn import LSTMStateTuple, LSTMCell
from chatbot.components.base._rnn import RNN, SimpleAttentionWrapper
from utils import io_utils


class Decoder(RNN):
    """Dynamic decoding (base) class that supports both training and inference without
       requiring superfluous helper objects. With simple boolean parameters,
       handles the decoder sub-graph construction dynamically in its entirety.
    """

    def __init__(self,
                 base_cell,
                 encoder_outputs,
                 state_size,
                 vocab_size,
                 embed_size,
                 dropout_prob,
                 num_layers,
                 temperature,
                 max_seq_len,
                 state_wrapper=None):
        """
        Args:
            base_cell: (str) name of RNNCell class for underlying cell.
            state_size: number of units in underlying rnn cell.
            vocab_size: dimension of output space for projections.
            embed_size: dimension size of word-embedding space.
            dropout_prob: probability of a node being dropped.
            num_layers: how many cells to include in the MultiRNNCell.
            temperature: (float) determines randomness of outputs/responses.
                - Some notable values (to get some intuition):
                  - t -> 0: outputs approach simple argmax.
                  - t = 1: same as sampling from softmax distribution over
                    outputs, interpreting the softmax outputs as from a
                    multinomial (probability) distribution.
                  - t -> inf: outputs approach uniform random distribution.
            state_wrapper: allow states to store their wrapper class. See the
                wrapper method docstring below for more info.
        """

        self.encoder_outputs = encoder_outputs
        if state_wrapper is None and base_cell == 'LSTMCell':
            state_wrapper = LSTMStateTuple

        super(Decoder, self).__init__(
            base_cell=base_cell,
            state_size=state_size,
            embed_size=embed_size,
            dropout_prob=dropout_prob,
            num_layers=num_layers,
            state_wrapper=state_wrapper)

        self.temperature = temperature
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        with tf.variable_scope('projection_tensors'):
            w = tf.get_variable(
                name="w",
                shape=[state_size, vocab_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                name="b",
                shape=[vocab_size],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
            self._projection = (w, b)

    def __call__(self,
                 inputs,
                 is_chatting,
                 loop_embedder,
                 cell,
                 initial_state=None):
        """Run the inputs on the decoder.

        If we are chatting, then conduct dynamic sampling, which is the process
        of generating a response given inputs == GO_ID.

        Args:
            inputs: Tensor with shape [batch_size, max_time, embed_size].
                For training, inputs are the 'to' sentence tokens (embedded).
                For chatting, first input is <GO> and thereafter, the input is
                the bot's previous output (looped around through embedding).
            initial_state: Tensor with shape [batch_size, state_size].
            is_chatting: (bool) Determines how we retrieve the outputs and the
                         returned Tensor shape.
            loop_embedder: required if is_chatting==True.
                           Embedder instance needed to feed decoder outputs
                           as next inputs.

        Returns:
            outputs: if not chatting, tensor of shape
                [batch_size, max_time, vocab_size]. Otherwise, tensor of
                response IDs with shape [batch_size, max_time].
            state:   if not is_chatting, tensor of shape
                [batch_size, state_size]. Otherwise, None.
        """

        self.rnn = tf.make_template('decoder_rnn',
                                    tf.nn.dynamic_rnn,
                                    cell=cell,
                                    dtype=tf.float32)

        outputs, state = self.rnn(inputs=inputs,
                                  initial_state=initial_state)

        if not is_chatting:
            return outputs, state

        if loop_embedder is None:
            raise ValueError(
                "Loop function required to feed outputs as inputs.")

        def body(response, state):
            """Input callable for tf.while_loop. See below."""
            tf.get_variable_scope().reuse_variables()
            decoder_input = loop_embedder(tf.reshape(response[-1], (1, 1)),
                                          reuse=True)

            outputs, state = self.rnn(inputs=decoder_input,
                                      initial_state=state,
                                      sequence_length=[1])

            next_id = self.sample(self.apply_projection(outputs))
            response = tf.concat([response, tf.stack([next_id])], axis=0)
            return response, state

        def cond(response, s):
            """Input callable for tf.while_loop. See below."""
            return tf.logical_and(
                tf.not_equal(response[-1], io_utils.EOS_ID),
                tf.less_equal(tf.size(response), self.max_seq_len))

        # Project to full output state during inference time.
        # Note: "outputs" at this point, at this exact line, is technically just
        # a single output: the bot's first response token.
        outputs = self.apply_projection(outputs)
        # Begin the process of building the list of output tokens.
        response = tf.stack([self.sample(outputs)])
        # Reshape is needed so the while_loop ahead knows the shape of response.
        # The comma after the 1 is intentional, it forces tf to believe us.
        response = tf.reshape(response, [1,], name='response')
        tf.get_variable_scope().reuse_variables()

        # ============== BEHOLD: The tensorflow while loop. ==================
        # This allows us to sample dynamically. It also makes me happy!
        # -- Repeat 'body' while the 'cond' returns true.
        # -- 'cond': callable returning a boolean scalar tensor.
        # -- 'body': callable returning a tuple of tensors of same
        #            arity as loop_vars.
        # -- 'loop_vars': tuple of tensors that is passed to 'cond' and 'body'.
        response, _ = tf.while_loop(
            cond, body, (response, state),
            shape_invariants=(tf.TensorShape([None]), cell.shape),
            back_prop=False)
        # =============== FAREWELL: The tensorflow while loop. =================

        outputs = tf.expand_dims(response, 0)
        return outputs, None

    def apply_projection(self, outputs, scope=None):
        """Defines & applies the affine transformation from state space
        to output space.

        Args:
            outputs: Tensor of shape [batch_size, max_time, state_size]
                returned by tf dynamic_rnn.
            scope: (optional) variable scope for any created here.

        Returns:
            Tensor of shape [batch_size, max_time, vocab_size] representing the
            projected outputs.
        """

        with tf.variable_scope(scope, "proj_scope", [outputs]):

            # Swap 1st and 2nd indices to match expected input of map_fn.
            seq_len = tf.shape(outputs)[1]
            st_size = tf.shape(outputs)[2]
            time_major_outputs = tf.reshape(outputs, [seq_len, -1, st_size])

            # Project batch at single timestep from state space to output space.
            def proj_op(batch):
                return tf.matmul(batch, self._projection[0]) + self._projection[1]

            # Get projected output states;
            # 3D Tensor with shape [batch_size, seq_len, ouput_size].
            projected_state = tf.map_fn(proj_op, time_major_outputs)
        return tf.reshape(projected_state, [-1, seq_len, self.vocab_size])

    def sample(self, projected_output):
        """Return integer ID tensor representing the sampled word.
        
        Args:
            projected_output: Tensor [1, 1, state_size], representing a single
                decoder timestep output. 
        """
        # TODO: We really need a tf.control_dependencies check here (for rank).
        with tf.name_scope('decoder_sampler', values=[projected_output]):

            # Protect against extra size-1 dimensions; grab the 1D tensor
            # of size state_size.
            logits = tf.squeeze(projected_output)
            if self.temperature < 0.02:
                return tf.argmax(logits, axis=0)

            # Convert logits to probability distribution.
            probabilities = tf.div(logits, self.temperature)
            projected_output = tf.div(
                tf.exp(probabilities),
                tf.reduce_sum(tf.exp(probabilities), axis=-1))

            # Sample 1 time from the probability distribution.
            sample_ID = tf.squeeze(
                tf.multinomial(tf.expand_dims(probabilities, 0), 1))
        return sample_ID

    def get_projection_tensors(self):
        """Returns the tuple (w, b) that decoder uses for projecting.
        Required as argument to the sampled softmax loss.
        """
        return self._projection


class BasicDecoder(Decoder):
    """Simple (but dynamic) decoder that is essentially just the base class."""

    def __call__(self,
                 inputs,
                 initial_state=None,
                 is_chatting=False,
                 loop_embedder=None,
                 cell=None):

        return super(BasicDecoder, self).__call__(
            inputs=inputs,
            initial_state=initial_state,
            is_chatting=is_chatting,
            loop_embedder=loop_embedder,
            cell=self.get_cell('decoder_cell'))


class AttentionDecoder(Decoder):
    """Dynamic decoder that applies an attention mechanism over the full
    sequence of encoder outputs. Using Bahdanau for now (may change).
    
    TODO: Luong's paper mentions that they only use the *top* layer of 
    stacked LSTMs for attention-related computation. Since currently I'm 
    only testing attention models with one-layer encoder/decoders, this
    isn't an issue. However, in a couple days I should revisit this.
    """

    def __init__(self,
                 encoder_outputs,
                 base_cell,
                 state_size,
                 vocab_size,
                 embed_size,
                 attention_mechanism='BahdanauAttention',
                 dropout_prob=1.0,
                 num_layers=1,
                 temperature=0.0,
                 max_seq_len=10):
        """We need to explicitly call the constructor now, so we can:
           - Specify we need the state wrapped in AttentionWrapperState.
           - Specify our attention mechanism (will allow customization soon).
        """

        super(AttentionDecoder, self).__init__(
            encoder_outputs=encoder_outputs,
            base_cell=base_cell,
            state_size=state_size,
            vocab_size=vocab_size,
            embed_size=embed_size,
            dropout_prob=dropout_prob,
            num_layers=num_layers,
            temperature=temperature,
            max_seq_len=max_seq_len,
            state_wrapper=AttentionWrapperState)

        _mechanism = getattr(tf.contrib.seq2seq, attention_mechanism)
        self.attention_mechanism = _mechanism(num_units=state_size,
                                              memory=encoder_outputs)
        self.output_attention = True

    def __call__(self,
                 inputs,
                 initial_state=None,
                 is_chatting=False,
                 loop_embedder=None,
                 cell=None):
        """
        The only modifcation to the superclass is we pass in our own
        cell that is wrapped with a custom attention class (specified in
        base/_rnn.py). It is mostly the same as tensorflow's, but with minor
        tweaks so that it could easily hang out with the other components of
        the project.
        """

        if cell is None:
            cell = self.get_cell('attn_cell', initial_state)

        return super(AttentionDecoder, self).__call__(
            inputs=inputs,
            is_chatting=is_chatting,
            loop_embedder=loop_embedder,
            cell=cell)

    def get_cell(self, name, initial_state):
        # Get the simple underlying cell first.
        cell = super(AttentionDecoder, self).get_cell(name)
        # Return the normal cell wrapped to support attention.
        return SimpleAttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            initial_cell_state=initial_state)


