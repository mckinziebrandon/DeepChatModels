"""Sequence-to-sequence model with an attention mechanism."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Standard python imports.
import random
# ML/DL-specific imports.
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq
from tensorflow.contrib.legacy_seq2seq import model_with_buckets
# User-defined imports.
from utils import *


class Chatbot(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder.
    """

    def __init__(self,
                 config,
                 buckets,
                 vocab_size=40000,
                 layer_size=512,
                 num_layers=3,
                 max_gradient=5.0,
                 batch_size=64,
                 learning_rate=0.5,
                 lr_decay=0.98,
                 num_softmax_samp=512,
                 is_decoding=False,
                 dataset_name = "ubuntu"):
        """Create the model.

        Args:
          vocab_size:   size of the source vocabulary.
          buckets:      a list of pairs (I, O), where I (O) specifies maximum input (output) length
                        that will be processed in that bucket
          layer_size:   number of units in each layer of the model.
          num_layers:   number of layers in the model.
          max_gradient: gradients will be clipped to maximally this norm.
          batch_size:   the size of the batches used during training;
          learning_rate:    learning rate to start with.
          lr_decay:         decay learning rate by this much when needed.
          num_softmax_samp: number of samples for sampled softmax.
          is_decoding:      if set, we do not construct the backward pass in the model.
          config: TODO
        """

        print("Beginning model construction . . . ")

        # =====================================================================================================
        # Store instance variables.
        # =====================================================================================================

        self.vocab_size = vocab_size
        self.buckets        = buckets
        self.batch_size     = batch_size
        self.learning_rate  = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay_op    = self.learning_rate.assign(learning_rate * lr_decay)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)
        self.dataset_name   = dataset_name
        self.config = config

        # =====================================================================================================
        # Define basic components: cell(s) state, encoder, decoder.
        # =====================================================================================================

        # Create the internal (potentially multi-layer) cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(layer_size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        else:
            cell = single_cell()

        # Feeds for encoder inputs.
        max_input_length    = buckets[-1][0]
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i))
                               for i in range(max_input_length)]

        # Feeds & weights for decoder inputs.
        max_output_length   = buckets[-1][1] + 1  # because we always append EOS after
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i))
                               for i in range(max_output_length)]
        self.target_weights = [tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i))
                               for i in range(max_output_length)]

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # Use sampled softmax if (1) user wants to: [num_softmax_samp>0],
        # and (2) it makes sense to do so: [num_softmax_samp<vocab_size].
        if 0 < num_softmax_samp < self.vocab_size:
            softmax_loss_function, output_projection = Chatbot._sampled_softmax_loss(
                                                        num_softmax_samp, layer_size, self.vocab_size)
        else:
            softmax_loss_function, output_projection = None, None

        # =====================================================================================================
        # Combine the components to construct desired model architecture.
        # =====================================================================================================

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                               num_encoder_symbols=vocab_size,
                                               num_decoder_symbols=vocab_size,
                                               embedding_size=layer_size, output_projection=output_projection,
                                               feed_previous=do_decode, dtype=tf.float32)

        # Note: if softmax_loss_func is None, model_with_buckets will default to standard softmax.
        self.outputs, self.losses = model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                       targets, self.target_weights,
                                                       buckets, lambda x, y: seq2seq_f(x, y, is_decoding),
                                                       softmax_loss_function=softmax_loss_function)

        # If decoding, append projection to true output to the model.
        if is_decoding and output_projection is not None:
            self.outputs = Chatbot._get_projections(len(buckets), self.outputs, output_projection)

        # =====================================================================================================
        # Configure training process (backward pass).
        # =====================================================================================================

        params = tf.trainable_variables()
        if not is_decoding:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            print("Looping over", len(buckets), "buckets.")
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        print("Creating saver and exiting . . . ")
        self.saver = tf.train.Saver(tf.global_variables())


    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
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
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(fetches=output_feed, feed_dict=input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

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

    def train(self, max_train_samples, data_dir, max_steps=10000):
        """ Train chatbot using 1-on-1 ubuntu dialogue corpus. """
        import chatbot.train as train
        self.sess = self._create_session()
        self._setup_parameters()
        train.train(self)

    def decode(self):
        import chatbot.decode as decode
        self.sess = self._create_session()
        self._setup_parameters()
        decode.decode(self)

    def _setup_parameters(self):
        # Check if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
        # If we can't, then just re-initialize model with fresh params.
        print("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(self.config.train_dir)
        if checkpoint_state and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
            print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())


    def _create_session(self):
        return tf.Session()

    @staticmethod
    def _sampled_softmax_loss(num_samples: int, hidden_size: int, vocab_size: int):
        """TODO
        :param num_samples:     (context: importance sampling) size of subset of outputs for softmax.
        :param hidden_size:     number of units in the individual recurrent states.
        :param vocab_size: number of unique output words.
        :return: sampled_loss, output_projection
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
