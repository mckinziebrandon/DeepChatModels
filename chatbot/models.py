"""Sequence-to-sequence model with an attention mechanism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard python imports.
import os
import random
from pathlib import Path

# ML/DL-specific imports.
import numpy as np
import tensorflow as tf
# Chatbot class.
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq import model_with_buckets
# Just in case (temporary)
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# User-defined imports.
from utils import data_utils
from chatbot._train import train
from chatbot._decode import decode


class Model(object):
    """Abstract model class with rudimentary implementations."""

    def __init__(self,
                 buckets,
                 vocab_size=40000,
                 batch_size=64,
                 learning_rate=0.5,
                 lr_decay=0.98,
                 is_decoding=False):
        self.sess           = tf.Session()
        self.is_decoding    = is_decoding
        self.batch_size     = batch_size
        self.buckets        = buckets
        self.vocab_size = vocab_size

        self.learning_rate  = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay_op    = self.learning_rate.assign(learning_rate * lr_decay)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)
        # Question: WHERE DO I PUT YOU :(
        self.saver          = tf.train.Saver(tf.global_variables())

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.
            encoder_inputs[i]       == [words in sentences[i]], where 0 <  i < batch_size.
            batch_encoder_inputs[i] == list(i'th wordID over all batch sentences)

        Args:
          data: tuple of len(self.buckets). data[bucket_id] == [source_ids, target_ids]
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad= [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

        # Define some small helper functions before we re-index & weight.
        def inputs_to_unit(uid, inputs):
            """ Return re-indexed version of inputs array. Description in params below.
            :param uid: index identifier for input timestep/unit/node of interest.
            :param inputs:  single batch of data; inputs[i] is i'th sentence.
            :return:        re-indexed version of inputs as numpy array.
            """
            return np.array([inputs[i][uid] for i in range(self.batch_size)], dtype=np.int32)

        batch_encoder_inputs = [inputs_to_unit(i, encoder_inputs) for i in range(encoder_size)]
        batch_decoder_inputs = [inputs_to_unit(i, decoder_inputs) for i in range(decoder_size)]
        batch_weights        = list(np.ones(shape=(decoder_size, self.batch_size), dtype=np.float32))

        # Set weight for the final decoder unit to 0.0 for all batches.
        for i in range(self.batch_size):
            batch_weights[-1][i] = 0.0

        # Also set any decoder-input-weights to 0 that have PAD as target decoder output.
        for unit_id in range(decoder_size - 1):
            ids_with_pad_target = [b for b in range(self.batch_size)
                                   if decoder_inputs[b][unit_id+1] == data_utils.PAD_ID]
            batch_weights[unit_id][ids_with_pad_target] = 0.0


        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
    def restore(self, meta_name):
        """The exact order as seen in tf tutorials:"""
        raise NotImplemented
        #with self.sess as sess:
            #self.saver = tf.train.import_meta_graph(meta_name)
            #checkpoint_state  = tf.train.get_checkpoint_state(self.config.ckpt_dir)
            #if not checkpoint_state:
            #    raise RuntimeError("Can't find ckpt.")
            #self.saver.restore(sess, checkpoint_state.model_checkpoint_path)
            #self.learning_rate = tf.get_collection("learning_rate")[0]
            #self.losses = tf.get_collection("losses")
            #self.outputs = [tf.get_collection("outputs{}".format(b)) for b in range(len(buckets))]

    def save(self, chatbot):
        raise NotImplemented
        #self.saver = tf.train.Saver(tf.global_variables())
        #tf.add_to_collection("learning_rate", self.learning_rate)
        #if self.is_decoding and chatbot.output_proj is not None:
        #    dec_outputs = Chatbot._get_projections(len(chatbot.buckets), self.outputs, chatbot.output_proj)
        #else:
        #    dec_outputs = self.outputs
        #for b in range(len(chatbot.buckets)):
        #    for o in dec_outputs[b]:
        #        tf.add_to_collection("outputs{}".format(b), o)
        #for b in range(len(chatbot.buckets)):
        #    tf.add_to_collection("losses", self.losses[b])

    def setup_parameters(self, config):
        """Either restore model parameters or create fresh ones.
            - Checks if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
            - If we can't, then just re-initialize model with fresh params.
        """
        print("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(config.ckpt_dir)
        # Note: If you want to prevent from loading models trained on different dataset,
        # you should store them in their own out/dataname folder, and pass that as the ckpt_dir to config.
        if checkpoint_state and not config.reset_model \
                and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
            print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            # Clear output dir contents.
            os.popen('rm -rf out/* && mkdir -p out/logs')
            # Run initializer operation after we've fully constructed model & launched it in a sess.
            self.sess.run(tf.global_variables_initializer())
        self.file_writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)


class Chatbot(Model):
    """Sequence-to-sequence model with attention and for multiple buckets.

    The input-to-output path can be thought of (on a high level) as follows:
        1. Inputs:      Batches of integer lists, where each integer is a word ID to a pre-defined vocabulary.
        2. Embedding:   each input integer is mapped to an embedding vector.
                        Each embedding vector is of length 'layer_size', an argument to __init__.
                        The encoder and decoder have their own distinct embedding spaces.
        3. Encoding:    The embedded batch vectors are fed to a multi-layer cell containing GRUs.
        4. Attention:   At each timestep, the output of the multi-layer cell is saved, so that
                        the decoder can access them in the manner specified in the paper on
                        jointly learning to align and translate. (should give a link to paper...)
        5. Decoding:    The decoder, the same type of embedded-multi-layer cell as the encoder, is initialized
                        with the last output of the encoder, the "context". Thereafter, we either feed it
                        a target sequence (when training) or we feed its previous output as its next input (chatting).
    """

    def __init__(self,
                 buckets: list,
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

        print("Beginning model construction . . . ")
        super(Chatbot, self).__init__(buckets,
                                      vocab_size=vocab_size,
                                      batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      lr_decay=lr_decay,
                                      is_decoding=is_decoding)

        # ==============================================================================================
        # Define basic components: cell(s) state, encoder, decoder.
        # ==============================================================================================

        cell = Chatbot._get_cell(num_layers, layer_size)
        self.encoder_inputs = Chatbot._get_placeholder_list("encoder", buckets[-1][0])
        self.decoder_inputs = Chatbot._get_placeholder_list("decoder", buckets[-1][1] + 1)
        self.target_weights = Chatbot._get_placeholder_list("weight", buckets[-1][1] + 1, tf.float32)
        target_outputs = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # Determine whether to draw sample subset from output softmax or just use default tensorflow softmax.
        if 0 < num_softmax_samp < self.vocab_size:
            softmax_loss, output_proj = Chatbot._sampled_softmax_loss(num_softmax_samp, layer_size, self.vocab_size)
        else:
            softmax_loss, output_proj = None, None

        # ==============================================================================================
        # Combine the components to construct desired model architecture.
        # ==============================================================================================

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs):
            # Note: the returned function uses separate embeddings for encoded/decoded sets.
            #           Maybe try implementing with same embedding for both; makes more sense for our purposes.
            # Question: the outputs are projected to vocab_size NO MATTER WHAT.
            #           i.e. if output_proj is None, it uses its own OutputProjectionWrapper instead
            #           --> How does this affect our model?? A bit misleading imo.
            return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                               num_encoder_symbols=vocab_size, num_decoder_symbols=vocab_size,
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

        self.summaries = {}
        for i, loss in enumerate(self.losses):
            name = "loss{}".format(i)
            self.summaries[name] = tf.summary.scalar("loss{}".format(i), loss)

        # If decoding, append projection to true output to the model.
        if is_decoding and output_proj is not None:
            self.outputs = Chatbot._get_projections(len(buckets), self.outputs, output_proj)

        # ==============================================================================================
        # Configure training process (backward pass).
        # ==============================================================================================

        # Note: variables are trainable=True by default.
        params = tf.trainable_variables()
        if not is_decoding:
            self.gradient_norms = []
            # updates will store the parameter (S)GD updates.
            self.updates = []
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            print("Looping over", len(buckets), "buckets.")
            # TODO: Think about how this could optimized. There has to be a way.
            for b in range(len(buckets)):
                # Note: tf.gradients returns in form: gradients[i] == sum([dy/dx_i for y in self.losses[b]]).
                gradients = tf.gradients(self.losses[b], params)
                # Gradient clipping is actually extremely simple, it basically just
                # checks if L2Norm(gradients) > max_gradient, and if it is, it returns
                # (gradients / L2Norm(gradients)) * max_grad.
                # norm: literally just L2-norm of gradients.
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                self.gradient_norms.append(norm)
                self.updates.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                              global_step=self.global_step))

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """

        def check_input_length(actual, expected):
            if actual == expected: return
            raise ValueError("Length must be equal to the one in bucket,"
                             " %d != %d." % (actual, expected))

        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        check_input_length(len(encoder_inputs), encoder_size)
        check_input_length(len(decoder_inputs), decoder_size)
        check_input_length(len(target_weights), decoder_size)

        # Input feed: Associate the parameters given with the model variables.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # The decoder is actually 1 step longer than bucket size, so don't forget to add final step.
        input_feed[self.decoder_inputs[decoder_size].name] = np.zeros([self.batch_size], dtype=np.int32)

        # Fetches: the Operations/Tensors we want executed/evaluated during session.run(...).
        if not forward_only:
            fetches = [self.summaries["loss{}".format(bucket_id)],
                       self.updates[bucket_id],         # Update Op that does SGD.
                       self.gradient_norms[bucket_id],  # Gradient norm.
                       self.losses[bucket_id]]          # Loss for this batch.
            outputs = session.run(fetches=fetches, feed_dict=input_feed)
            return outputs[0], outputs[2], outputs[3], None  # summaries,  Gradient norm, loss, no outputs.
        else:
            fetches = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):       # Output logits.
                fetches.append(self.outputs[bucket_id][l])
            outputs = session.run(fetches=fetches, feed_dict=input_feed)
            return None, None, outputs[0], outputs[1:]  #No summary,  No gradient norm, loss, outputs.

    def train(self, dataset, train_config):
        """ Train chatbot. """
        super(Chatbot, self).setup_parameters(train_config)
        train(self, dataset, train_config)

    def decode(self, test_config):
        """ Create chat session between user & chatbot. """
        super(Chatbot, self).setup_parameters(test_config)
        decode(self, test_config)

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


class SimpleBot(Model):
    """Primitive implementation from scratch, for learning purposes.
            1. Inputs: same as Chatbot.
            2. Embedding: same as Chatbot.
            3. Encoder: Single GRUCell.
            4. Decoder: Single GRUCell.
    """

    def __init__(self,
                 max_seq_len = 20,
                 vocab_size=40000,
                 layer_size=512,
                 max_gradient=5.0,
                 batch_size=64,     # TODO: shouldn't be here -- training specific.
                 learning_rate=0.5,
                 lr_decay=0.98,
                 is_decoding=False):

        print("Beginning model construction . . . ")
        # SimpleBot allows user to not worry about making their own buckets.
        # SimpleBot does that for you.
        # SimpleBot cares.
        buckets = [(max_seq_len // 2,  max_seq_len // 2),
                        (max_seq_len, max_seq_len)]

        super(SimpleBot, self).__init__(buckets,
                                        vocab_size=vocab_size,
                                        batch_size=batch_size,
                                        learning_rate=learning_rate,
                                        lr_decay=lr_decay,
                                        is_decoding=is_decoding)

        # ==============================================================================================
        # Create placeholder lists for encoder/decoder sequences.
        # ==============================================================================================

        cell = tf.contrib.rnn.GRUCell(layer_size)
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder"+str(i)) for i in range(max_seq_len)]
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder"+str(i)) for i in range(max_seq_len+1)]
        self.target_weights = [tf.placeholder(tf.float32, shape=[None], name="weight"+str(i)) for i in range(max_seq_len+1)]

        # ====================================================================================
        # Before bucketing, need to define the underlying model(x, y) -> outputs, state(s).
        # ====================================================================================

        def seq2seq(encoder_inputs, decoder_inputs):
            """Builds basic encoder-decoder model and returns list of (2D) output tensors."""
            with tf.variable_scope("basic_rnn_seq2seq"):
                _, encoder_state = core_rnn.static_rnn(cell, encoder_inputs, dtype=tf.float32)
                with tf.variable_scope("rnn_decoder") as decoder_scope:
                    decoder_outputs = []
                    for i, dec_inp in enumerate(decoder_inputs):
                        if i > 0:
                            decoder_scope.reuse_variables()
                        output, encoder_state = cell(dec_inp, encoder_state)
                        decoder_outputs.append(output)
                return decoder_outputs

        # ====================================================================================
        # Now we can build a simple bucketed seq2seq model.
        # ====================================================================================

        losses  = []
        outputs = []
        values  = self.encoder_inputs + self.decoder_inputs + self.decoder_inputs
        with tf.name_scope("simple_bucket_model", values):
            for idx_b, bucket in enumerate(self.buckets):
                # Reminder: you should never explicitly set reuse=False. It's a no-no.
                with tf.variable_scope(tf.get_variable_scope(), reuse=True if idx_b > 0 else None):
                    # The outputs for this bucket are defined entirely by the seq2seq function.
                    outputs.append(seq2seq(self.encoder_inputs[:bucket[0]],
                                           self.decoder_inputs[:bucket[1]]))
                    # Target outputs are just the inputs time-shifted by 1.
                    target_outputs = [self.decoder_inputs[i + 1]
                                      for i in range(len(self.decoder_inputs) - 1)]
                    # Compute loss by comparing outputs and target outputs.
                    losses.append(self._simple_loss(outputs[-1],
                                                    target_outputs[:bucket[1]],
                                                    self.target_weights))


    def _simple_loss(self, logits, targets, weights):
        """Compute weighted cross-entropy loss on softmax(logits)."""
        # Note: name_scope only affects names of ops, while variable_scope affects both ops AND variables.
        with tf.name_scope("simple_loss", values=logits+targets):
            log_perplexities = []
            for l, t, w in zip(logits, targets, weights):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=l)
                log_perplexities.append(cross_entropy * w)
        # Reduce via elementwise-sum.
        log_perplexities = tf.add_n(log_perplexities)
        # Get weighted-averge by dividing by sum of the weights.
        log_perplexities /= tf.add_n(weights) + 1e-12
        return tf.reduce_sum(log_perplexities) / tf.cast(self.batch_size, tf.float32)






















