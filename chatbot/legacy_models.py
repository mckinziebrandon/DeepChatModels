"""Sequence-to-sequence models."""

# TODO: Figure out what to do with these.
# They worked in tensorflow 0.12, but not in 1.0, and for good reason.
# Since there is no immediately obvious way of converting these to the better,
# dynamic models available in tf r1.0, I'm going to make them from scratch in
# the file dynamic_models.py for now.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ML/DL-specific imports.
import numpy as np
import tensorflow as tf
# ChatBot class.
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq import model_with_buckets
# Just in case (temporary)
from tensorflow.contrib.rnn.python.ops import core_rnn
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

    def __init__(self,
                 buckets,
                 log_dir = "out/logs",
                 vocab_size=40000,
                 layer_size=512,
                 num_layers=3,
                 max_gradient=5.0,
                 batch_size=64,     # TODO: shouldn't be here -- training specific.
                 learning_rate=0.5,
                 lr_decay=0.98,
                 num_softmax_samp=512,
                 is_decoding=False):
        """Create the model.

        Args:
            vocab_size: number of unique tokens in the dataset vocabulary.
            buckets: a list of pairs (I, O), where I (O) specifies maximum input (output) length
                     that will be processed in that bucket.
            layer_size: number of units in each recurrent layer (contained within the model cell).
            num_layers: number of recurrent layers in the model's cell state.
            max_gradient: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
            learning_rate: learning rate to start with.
            lr_decay: decay learning rate by this much when needed.
            num_softmax_samp: number of samples for sampled softmax.
            is_decoding: if True, don't build backward pass.
        """

        # ==============================================================================================
        # Define basic components: cell(s) state, encoder, decoder.
        # ==============================================================================================

        cell = ChatBot._get_cell(num_layers, layer_size)
        self.encoder_inputs = ChatBot._get_placeholder_list("encoder", buckets[-1][0])
        self.decoder_inputs = ChatBot._get_placeholder_list("decoder", buckets[-1][1] + 1)
        self.target_weights = ChatBot._get_placeholder_list("weight", buckets[-1][1] + 1, tf.float32)
        target_outputs = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # Determine whether to draw sample subset from output softmax or just use default tensorflow softmax.
        if 0 < num_softmax_samp < vocab_size:
            softmax_loss, output_proj = ChatBot._sampled_softmax_loss(num_softmax_samp, layer_size, vocab_size)
        else:
            softmax_loss, output_proj = None, None

        # ==============================================================================================
        # Combine the components to construct desired model architecture.
        # ==============================================================================================

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            # Note: the returned function uses separate embeddings for encoded/decoded sets.
            #           Maybe try implementing same embedding for both.
            # Question: the outputs are projected to vocab_size NO MATTER WHAT.
            #           i.e. if output_proj is None, it uses its own OutputProjectionWrapper instead
            #           --> How does this affect our model?? A bit misleading imo.
            with tf.variable_scope("seq2seq2_f"):
                return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                                   num_encoder_symbols=vocab_size,
                                                   num_decoder_symbols=vocab_size,
                                                   embedding_size=layer_size,
                                                   output_projection=output_proj,
                                                   feed_previous=is_decoding,
                                                   dtype=tf.float32)

        # Note that self.outputs and self.losses are lists of length len(buckets).
        # This allows us to identify which outputs/losses to compute given a particular bucket.
        # Furthermore, \forall i < j, len(self.outputs[i])  < len(self.outputs[j]). Same goes for losses.
        self.outputs, self.losses = model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                       target_outputs, self.target_weights,
                                                       buckets, lambda x, y: seq2seq_f(x, y),
                                                       softmax_loss_function=softmax_loss)

        # If decoding, append projection to true output to the model.
        if is_decoding and output_proj is not None:
            self.outputs = ChatBot._get_projections(len(buckets), self.outputs, output_proj)

        with tf.variable_scope("summaries"):
            self.summaries = {}
            for i, loss in enumerate(self.losses):
                name = "loss{}".format(i)
                self.summaries[name] = tf.summary.scalar("loss{}".format(i), loss)

        super(ChatBot, self).__init__(buckets,
                                      ckpt_dir=log_dir,
                                      vocab_size=vocab_size,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      lr_decay=lr_decay,
                                      is_decoding=is_decoding)

        super(ChatBot, self).compile(self.losses, max_gradient)

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
            return outputs[0], None, outputs[2], None
        else:
            fetches = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):       # Output logits.
                fetches.append(self.outputs[bucket_id][l])
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return None, None, outputs[0], outputs[1:]

    @staticmethod
    def _sampled_softmax_loss(num_samples: int, hidden_size: int, vocab_size: int):
        """Defines the samples softmax loss op and the associated output projection.
        Args:
            num_samples:     (context: importance sampling) size of subset of outputs for softmax.
            hidden_size:     number of units in the individual recurrent states.
            vocab_size: number of unique output words.
        Returns:
            sampled_loss, output_projection
            - function: sampled_loss(labels, inputs)
            - output_projection: transformation to full vocab space, applied to decoder output.
        """

        assert(0 < num_samples < vocab_size)

        # Define the standard affine-softmax transformation from hidden_size -> vocab_size.
        # True output (for a given bucket) := tf.matmul(decoder_out, w) + b
        w_t = tf.get_variable("proj_w", [vocab_size, hidden_size], dtype=tf.float32)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [vocab_size], dtype=tf.float32)
        output_projection = (w, b)

        def sampled_loss(labels, inputs):
            # QUESTION: (1) Why reshape? (2) Explain the math (sec 3 of paper).
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
        """Apply projection operator to unprojected_vals, a tuple of length num_buckets.

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
    def _get_cell(num_layers, layer_size):
        # Create the internal (potentially multi-layer) cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(layer_size)
        if num_layers > 1:
            return tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        else:
            return single_cell()

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
            3. Encoder: Single GRUCell.
            4. Decoder: Single GRUCell.
    """

    def __init__(self,
                 log_dir = "out/logs",
                 max_seq_len = 30,
                 vocab_size=40000,
                 layer_size=512,
                 max_gradient=5.0,
                 batch_size=64,     # TODO: shouldn't be here -- training specific.
                 learning_rate=0.5,
                 lr_decay=0.98,
                 is_decoding=False):

        # SimpleBot allows user to not worry about making their own buckets.
        # SimpleBot does that for you. SimpleBot cares.
        buckets = [(max_seq_len // 2,  max_seq_len // 2), (max_seq_len, max_seq_len)]

        # ==========================================================================================
        # Create placeholder lists for encoder/decoder sequences.
        # ==========================================================================================

        # Base of cell: GRU.
        #base_cell = tf.contrib.rnn.GRUCell(layer_size)
        with tf.variable_scope("placeholders"):
            self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder"+str(i))
                                   for i in range(max_seq_len)]
            self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder"+str(i))
                                   for i in range(max_seq_len+1)]
            self.target_weights = [tf.placeholder(tf.float32, shape=[None], name="weight"+str(i))
                                   for i in range(max_seq_len+1)]

        # ====================================================================================
        # Before bucketing, need to define the underlying model(x, y) -> outputs, state(s).
        # ====================================================================================

        def seq2seq(encoder_inputs, decoder_inputs):
            """Builds basic encoder-decoder model and returns list of (2D) output tensors."""
            with tf.variable_scope("seq2seq"):
                encoder_cell = tf.contrib.rnn.GRUCell(layer_size)
                encoder_cell = tf.contrib.rnn.EmbeddingWrapper(encoder_cell, vocab_size, layer_size)
                # Encoder(raw_inputs) -> Embed(raw_inputs) -> [be an RNN] -> encoder state.
                _, encoder_state = core_rnn.static_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)
                with tf.variable_scope("decoder") as decoder_scope:

                    embedding = tf.get_variable("embedding", [vocab_size, layer_size])
                    decoder_cell = tf.contrib.rnn.GRUCell(layer_size)
                    decoder_cell = tf.contrib.rnn.EmbeddingWrapper(decoder_cell, vocab_size, layer_size)
                    decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size)

                    decoder_outputs = []
                    prev = None
                    decoder_state = None

                    loop_function = lambda x : embedding_ops.embedding_lookup(embedding, tf.argmax(x, 1))

                    for i, dec_inp in enumerate(decoder_inputs):

                        #if is_decoding and prev is not None:
                        #    dec_inp = loop_function(prev)

                        if i == 0:
                            #decoder_scope.reuse_variables()
                            output, decoder_state = decoder_cell(dec_inp, encoder_state)
                        else:
                            decoder_scope.reuse_variables()
                            output, decoder_state = decoder_cell(dec_inp, decoder_state)

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
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if idx_b > 0 else None):
                    # The outputs for this bucket are defined entirely by the seq2seq function.
                    self.outputs.append(seq2seq(self.encoder_inputs[:bucket[0]],
                                           self.decoder_inputs[:bucket[1]]))
                    # Target outputs are just the inputs time-shifted by 1.
                    target_outputs = [self.decoder_inputs[i + 1]
                                      for i in range(len(self.decoder_inputs) - 1)]
                    # Compute loss by comparing outputs and target outputs.
                    self.losses.append(SimpleBot._simple_loss(batch_size,
                                                              self.outputs[-1],
                                                    target_outputs[:bucket[1]],
                                                    self.target_weights[:bucket[1]]))

        with tf.variable_scope("summaries"):
            self.summaries = {}
            for i, loss in enumerate(self.losses):
                name = "loss{}".format(i)
                self.summaries[name] = tf.summary.scalar("loss{}".format(i), loss)

        # Let superclass handle the boring stuff :)
        super(SimpleBot, self).__init__(buckets,
                                        ckpt_dir="out",
                                        vocab_size=vocab_size,
                                        batch_size=batch_size,
                                        learning_rate=learning_rate,
                                        lr_decay=lr_decay,
                                        is_decoding=is_decoding)

        super(SimpleBot, self).compile(self.losses, max_gradient)

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
            return outputs[0], None, outputs[3], None  # summaries,  No gradient norm, loss, no outputs.
        else:
            fetches = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):       # Output logits.
                fetches.append(self.outputs[bucket_id][l])
            outputs = self.sess.run(fetches=fetches, feed_dict=input_feed)
            return None, None, outputs[0], outputs[1:]  #No summary,  No gradient norm, loss, outputs.
