"""Sequence-to-sequence models."""

# EDIT: Modified inheritance strucutre (see _models.py) so these *should* work again.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq import model_with_buckets
#from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import embedding_ops
from chatbot._models import BucketModel


class ChatBot(BucketModel):
    """Sequence-to-sequence model with attention and for multiple buckets.

    The input-to-output path can be thought of (on a high level) as follows:
        1. Inputs:      Batches of integer lists, where each integer is a
                        word ID to a pre-defined vocabulary.
        2. Embedding:   each input integer is mapped to an embedding vector.
                        Each embedding vector is of length 'layer_size', an argument to __init__.
                        The encoder and decoder have their own distinct embedding spaces.
        3. Encoding:    The embedded batch vectors are fed to a multi-layer cell containing GRUs.
        4. Attention:   At each timestep, the output of the multi-layer cell is saved, so that
                        the decoder can access them in the manner specified in the paper on
                        jointly learning to align and translate. (should give a link to paper...)
        5. Decoding:    The decoder, the same type of embedded-multi-layer cell
                        as the encoder, is initialized with the last output of the encoder,
                        the "context". Thereafter, we either feed it a target sequence
                        (when training) or we feed its previous output as its next input (chatting).
    """

    def __init__(self, buckets, dataset, params):

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('ChatBotLogger')
        super(ChatBot, self).__init__(
            logger=logger,
            buckets=buckets,
            dataset=dataset,
            params=params)

        if len(buckets) > 1:
            self.log.error("ChatBot requires len(buckets) be 1 since tensorflow's"
                           " model_with_buckets function is now deprecated and BROKEN. The only"
                           "workaround is ensuring len(buckets) == 1. ChatBot apologizes."
                           "ChatBot also wishes it didn't have to be this way. "
                           "ChatBot is jealous that DynamicBot does not have these issues.")
            raise ValueError("Not allowed to pass buckets with len(buckets) > 1.")

        # ==========================================================================================
        # Define basic components: cell(s) state, encoder, decoder.
        # ==========================================================================================

        #cell =  tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(state_size)for _ in range(num_layers)])
        cell = tf.contrib.rnn.GRUCell(self.state_size)
        self.encoder_inputs = ChatBot._get_placeholder_list("encoder", buckets[-1][0])
        self.decoder_inputs = ChatBot._get_placeholder_list("decoder", buckets[-1][1] + 1)
        self.target_weights = ChatBot._get_placeholder_list("weight", buckets[-1][1] + 1, tf.float32)
        target_outputs = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # If specified, sample from subset of full vocabulary size during training.
        softmax_loss, output_proj = None, None
        if 0 < self.num_samples < self.vocab_size:
            softmax_loss, output_proj = ChatBot._sampled_loss(self.num_samples,
                                                              self.state_size,
                                                              self.vocab_size)

        # ==========================================================================================
        # Combine the components to construct desired model architecture.
        # ==========================================================================================

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            # Note: the returned function uses separate embeddings for encoded/decoded sets.
            #           Maybe try implementing same embedding for both.
            # Question: the outputs are projected to vocab_size NO MATTER WHAT.
            #           i.e. if output_proj is None, it uses its own OutputProjectionWrapper instead
            #           --> How does this affect our model?? A bit misleading imo.
            #with tf.variable_scope(scope or "seq2seq2_f") as seq_scope:
            return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                               num_encoder_symbols=self.vocab_size,
                                               num_decoder_symbols=self.vocab_size,
                                               embedding_size=self.state_size,
                                               output_projection=output_proj,
                                               feed_previous=self.is_chatting,
                                               dtype=tf.float32)

        # Note that self.outputs and self.losses are lists of length len(buckets).
        # This allows us to identify which outputs/losses to compute given a particular bucket.
        # Furthermore, \forall i < j, len(self.outputs[i])  < len(self.outputs[j]). (same for loss)
        self.outputs, self.losses = model_with_buckets(
            self.encoder_inputs, self.decoder_inputs,
            target_outputs, self.target_weights,
            buckets, seq2seq_f,
            softmax_loss_function=softmax_loss)

        # If decoding, append _projection to true output to the model.
        if self.is_chatting and output_proj is not None:
            self.outputs = ChatBot._get_projections(len(buckets), self.outputs, output_proj)

        with tf.variable_scope("summaries"):
            self.summaries = {}
            for i, loss in enumerate(self.losses):
                name = "loss{}".format(i)
                self.summaries[name] = tf.summary.scalar("loss{}".format(i), loss)

    def step(self, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False):
        """Run a step of the model.

        Args:
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.

        Returns:
            [summary, gradient_norms, loss, outputs]
        """

        encoder_size, decoder_size = self.buckets[bucket_id]
        super(ChatBot, self).check_input_lengths(
            [encoder_inputs, decoder_inputs, target_weights],
            [encoder_size, decoder_size, decoder_size])

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        input_feed[self.decoder_inputs[decoder_size].name] = np.zeros([self.batch_size],
                                                                      dtype=np.int32)

        if not forward_only:  # Not just for decoding; also for validating in training.
            fetches = [self.summaries["loss{}".format(bucket_id)],
                       self.apply_gradients[bucket_id],  # Update Op that does SGD.
                       self.losses[bucket_id]]          # Loss for this batch.
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return outputs[0], None, outputs[2], None # Summary, no gradients, loss, outputs.
        else:
            fetches = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):       # Output logits.
                fetches.append(self.outputs[bucket_id][l])
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return None, None, outputs[0], outputs[1:] # No summary, no gradients, loss, outputs.

    @staticmethod
    def _sampled_loss(num_samples, hidden_size, vocab_size):
        """Defines the samples softmax loss op and the associated output _projection.
        Args:
            num_samples:     (context: importance sampling) size of subset of outputs for softmax.
            hidden_size:     number of units in the individual recurrent states.
            vocab_size: number of unique output words.
        Returns:
            sampled_loss, apply_projection
            - function: sampled_loss(labels, inputs)
            - apply_projection: transformation to full vocab space, applied to decoder output.
        """

        assert(0 < num_samples < vocab_size)

        # Define the standard affine-softmax transformation from hidden_size -> vocab_size.
        # True output (for a given bucket) := tf.matmul(decoder_out, w) + b
        w_t = tf.get_variable("proj_w", [vocab_size, hidden_size], dtype=tf.float32)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [vocab_size], dtype=tf.float32)
        output_projection = (w, b)

        def sampled_loss(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                    weights=w_t,
                    biases=b,
                    labels=labels,
                    inputs=inputs,
                    num_sampled=num_samples,
                    num_classes=vocab_size)

        return sampled_loss, output_projection

    @staticmethod
    def _get_projections(num_buckets, unprojected_vals, projection_operator):
        """Apply _projection operator to unprojected_vals, a tuple of length num_buckets.

        :param num_buckets:         the number of projections that will be applied.
        :param unprojected_vals:    tuple of length num_buckets.
        :param projection_operator: (in the mathematical meaning) tuple of shape unprojected_vals.shape[-1].
        :return: tuple of length num_buckets, with entries the same shape as entries in unprojected_vals, except for the last dimension.
        """
        projected_vals = unprojected_vals
        for b in range(num_buckets):
            projected_vals[b] = [tf.matmul(output, projection_operator[0]) + projection_operator[1]
                                 for output in unprojected_vals[b]]
        return projected_vals

    @staticmethod
    def _get_placeholder_list(name, length, dtype=tf.int32):
        """
        Args:
            name: prefix of name of each tf.placeholder list item, where i'th name is [name]i.
            length: number of items (tf.placeholders) in the returned list.
        Returns:
            list of tensorflow placeholder of dtype=tf.int32 and unspecified shape.
        """
        return [tf.placeholder(dtype, shape=[None], name=name+str(i)) for i in range(length)]


class SimpleBot(BucketModel):
    """Primitive implementation from scratch, for learning purposes.
            1. Inputs: same as ChatBot.
            2. Embedding: same as ChatBot.
            3. BasicEncoder: Single GRUCell.
            4. DynamicDecoder: Single GRUCell.
    """

    def __init__(self, dataset, params):

        # SimpleBot allows user to not worry about making their own buckets.
        # SimpleBot does that for you. SimpleBot cares.
        max_seq_len = dataset.max_seq_len
        buckets = [(max_seq_len // 2,  max_seq_len // 2), (max_seq_len, max_seq_len)]
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('SimpleBotLogger')
        super(SimpleBot, self).__init__(
            logger=logger,
            buckets=buckets,
            dataset=dataset,
            params=params)


        # ==========================================================================================
        # Create placeholder lists for encoder/decoder sequences.
        # ==========================================================================================

        with tf.variable_scope("placeholders"):
            self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder"+str(i))
                                   for i in range(self.max_seq_len)]
            self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder"+str(i))
                                   for i in range(self.max_seq_len+1)]
            self.target_weights = [tf.placeholder(tf.float32, shape=[None], name="weight"+str(i))
                                   for i in range(self.max_seq_len+1)]

        # ==========================================================================================
        # Before bucketing, need to define the underlying model(x, y) -> outputs, state(s).
        # ==========================================================================================

        def seq2seq(encoder_inputs, decoder_inputs, scope=None):
            """Builds basic encoder-decoder model and returns list of (2D) output tensors."""
            with tf.variable_scope(scope or "seq2seq"):
                encoder_cell = tf.contrib.rnn.GRUCell(self.state_size)
                encoder_cell = tf.contrib.rnn.EmbeddingWrapper(encoder_cell, self.vocab_size, self.state_size)
                # BasicEncoder(raw_inputs) -> Embed(raw_inputs) -> [be an RNN] -> encoder state.
                _, encoder_state = tf.contrib.rnn.static_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)
                with tf.variable_scope("decoder"):

                    def loop_function(x):
                        with tf.variable_scope("loop_function"):
                            params = tf.get_variable("embed_tensor", [self.vocab_size, self.state_size])
                            return embedding_ops.embedding_lookup(params, tf.argmax(x, 1))

                    _decoder_cell = tf.contrib.rnn.GRUCell(self.state_size)
                    _decoder_cell = tf.contrib.rnn.EmbeddingWrapper(_decoder_cell, self.vocab_size, self.state_size)
                    # Dear TensorFlow: you should replace the 'reuse' param in
                    # OutputProjectionWrapper with 'scope' and just do scope.reuse in __init__.
                    # sincerely, programming conventions.
                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                        _decoder_cell, self.vocab_size, reuse=tf.get_variable_scope().reuse)

                    decoder_outputs = []
                    prev = None
                    decoder_state = None

                    for i, dec_inp in enumerate(decoder_inputs):
                        if self.is_chatting and prev is not None:
                            dec_inp = loop_function(tf.reshape(prev, [1, 1]))
                        if i == 0:
                            output, decoder_state = decoder_cell(dec_inp, encoder_state,
                                                                 scope=tf.get_variable_scope())
                        else:
                            tf.get_variable_scope().reuse_variables()
                            output, decoder_state = decoder_cell(dec_inp, decoder_state,
                                                                 scope=tf.get_variable_scope())
                        decoder_outputs.append(output)
                return decoder_outputs

        # ====================================================================================
        # Now we can build a simple bucketed seq2seq model.
        # ====================================================================================

        self.losses  = []
        self.outputs = []
        values  = self.encoder_inputs + self.decoder_inputs + self.decoder_inputs
        with tf.name_scope("simple_bucket_model", values):
            for idx_b, bucket in enumerate(buckets):
                # Reminder: you should never explicitly set reuse=False. It's a no-no.
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if idx_b > 0 else None)\
                        as bucket_scope:
                    # The outputs for this bucket are defined entirely by the seq2seq function.
                    self.outputs.append(seq2seq(
                        self.encoder_inputs[:bucket[0]],
                        self.decoder_inputs[:bucket[1]],
                        scope=bucket_scope))
                    # Target outputs are just the inputs time-shifted by 1.
                    target_outputs = [self.decoder_inputs[i + 1]
                                      for i in range(len(self.decoder_inputs) - 1)]
                    # Compute loss by comparing outputs and target outputs.
                    self.losses.append(SimpleBot._simple_loss(self.batch_size,
                                                              self.outputs[-1],
                                                    target_outputs[:bucket[1]],
                                                    self.target_weights[:bucket[1]]))

        with tf.variable_scope("summaries"):
            self.summaries = {}
            for i, loss in enumerate(self.losses):
                name = "loss{}".format(i)
                self.summaries[name] = tf.summary.scalar("loss{}".format(i), loss)

    @staticmethod
    def _simple_loss(batch_size, logits, targets, weights):
        """Compute weighted cross-entropy loss on softmax(logits)."""
        # Note: name_scope only affects names of ops,
        # while variable_scope affects both ops AND variables.
        with tf.name_scope("simple_loss", values=logits+targets+weights):
            log_perplexities = []
            for l, t, w in zip(logits, targets, weights):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=l)
                log_perplexities.append(cross_entropy * w)
        # Reduce via elementwise-sum.
        log_perplexities = tf.add_n(log_perplexities)
        # Get weighted-averge by dividing by sum of the weights.
        log_perplexities /= tf.add_n(weights) + 1e-12
        return tf.reduce_sum(log_perplexities) / tf.cast(batch_size, tf.float32)

    def step(self, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False):
        """Run a step of the model.

        Args:
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.

        Returns:
            [summary, gradient_norms, loss, outputs]:
        """

        encoder_size, decoder_size = self.buckets[bucket_id]
        super(SimpleBot, self).check_input_lengths(
            [encoder_inputs, decoder_inputs, target_weights],
            [encoder_size, decoder_size, decoder_size])

        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        input_feed[self.decoder_inputs[decoder_size].name] = np.zeros([self.batch_size], dtype=np.int32)

        # Fetches: the Operations/Tensors we want executed/evaluated during session.run(...).
        if not forward_only: # Not just for decoding; also for validating in training.
            fetches = [self.summaries["loss{}".format(bucket_id)],
                       self.apply_gradients[bucket_id],  # Update Op that does SGD.
                       self.losses[bucket_id]]          # Loss for this batch.
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return outputs[0], None, outputs[2], None  # summaries,  No gradient norm, loss, no outputs.
        else:
            fetches = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):       # Output logits.
                fetches.append(self.outputs[bucket_id][l])
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return None, None, outputs[0], outputs[1:]  #No summary,  No gradient norm, loss, outputs.
