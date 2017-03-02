"""Sequence-to-sequence model with an attention mechanism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard python imports.
import random
import os

# ML/DL-specific imports.
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq import model_with_buckets

# User-defined imports.
from utils import data_utils
from utils import Config
from chatbot._train import _train
from chatbot._decode import _decode


class Chatbot(object):
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
                 batch_size=64,
                 learning_rate=0.5,
                 lr_decay=0.98,
                 num_softmax_samp=512,
                 is_decoding=False,
                 debug_mode=False):
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
        if debug_mode:
            print("[DEBUG] Layer size:", layer_size)
            print("[DEBUG] Number of layers:", num_layers)

        # ==============================================================================================
        # Store instance variables.
        # ==============================================================================================

        self.vocab_size     = vocab_size
        self.buckets        = buckets
        self.batch_size     = batch_size
        self.debug_mode = debug_mode

        self.learning_rate  = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay_op    = self.learning_rate.assign(learning_rate * lr_decay)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)

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
                                               embedding_size=layer_size, output_projection=output_proj,
                                               feed_previous=is_decoding, dtype=tf.float32)

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

        print("Creating saver and exiting . . . ")
        self.saver = tf.train.Saver(tf.global_variables())

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

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Don't forget the additional EOS only in **SELF**.decoder_inputs.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.summaries["loss{}".format(bucket_id)],
                           self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(fetches=output_feed, feed_dict=input_feed)
        if not forward_only:
            return outputs[0], outputs[2], outputs[3], None  # summaries,  Gradient norm, loss, no outputs.
        else:
            return None, None, outputs[0], outputs[1:]  #summary,  No gradient norm, loss, outputs.


    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        Clarification on variables below:
            encoder_inputs[i]       == list(all words in i'th batch sentence).
            batch_encoder_inputs[i] == list(i'th wordID over all batch sentences)

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
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
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Define some small helper functions before we re-index & weight.
        def inputs_to_unit(unit_id, inputs):
            """ Return re-indexed version of inputs array. Description in params below.
            :param unit_id: index identifier for input timestep/unit/node of interest.
            :param inputs:  array of length batch_size, where inputs[i]
                            is the sentence corresp. to i'th batch.
            :return:        re-indexed version of inputs as numpy array, where now indices are:
                            returned_arr[i] == list of input (words) to i'th unit/node/timestep over all batches.
            """
            return np.array([inputs[i][unit_id] for i in range(self.batch_size)], dtype=np.int32)

        def next_unit_is_pad(bid, uid):
            return decoder_inputs[bid][uid + 1] == data_utils.PAD_ID

        batch_encoder_inputs = [inputs_to_unit(i, encoder_inputs) for i in range(encoder_size)]
        batch_decoder_inputs = [inputs_to_unit(i, decoder_inputs) for i in range(decoder_size)]
        batch_weights        = list(np.ones(shape=(decoder_size, self.batch_size), dtype=np.float32))

        # Set weight for the final decoder unit to 0.0 for all batches.
        for i in range(self.batch_size):
            batch_weights[-1][i] = 0.0

        # Also set any decoder-input-weights to 0 that have PAD as target decoder output.
        for unit_id in range(decoder_size - 1):
            ids_with_pad_target = [i for i in range(self.batch_size) if next_unit_is_pad(i, unit_id)]
            batch_weights[unit_id][ids_with_pad_target] = 0.0

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def train(self, config: Config):
        """ Train chatbot. """

        self.sess = self._create_session()
        self.train_writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)
        self._setup_parameters(config)
        _train(self, config)

    def decode(self, config: Config):
        """ Create chat session between user & chatbot. """
        self.sess = self._create_session()
        self._setup_parameters(config)
        _decode(self, config)

    def _setup_parameters(self, config):
        # Check if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
        # If we can't, then just re-initialize model with fresh params.
        print("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(config.ckpt_dir)
        if checkpoint_state  and not config.reset_model \
                and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path) \
                and str(checkpoint_state.model_checkpoint_path).find(config.data_name) != -1:
            print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            # clear output dir contents.
            os.popen('mkdir -p out/logs')
            os.popen('mv out/logs .')
            os.popen('rm -rf out/*')
            os.popen('mv logs out/')
            # TODO: should probably create filewriter both here and after if statement, its actually less clunky...
            # Write all summaries to log_dir.
            self.sess.run(tf.global_variables_initializer())


    def _create_session(self):
        return tf.Session()

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
