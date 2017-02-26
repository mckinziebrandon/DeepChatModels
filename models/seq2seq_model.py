"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Standard python imports.
import random
import sys
# ML/DL-specific imports.
import tensorflow as tf
import numpy as np
# User-defined imports.
import data_utils as data_utils


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder.
    """

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 lr_decay,
                 num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
          source_vocab_size:    size of the source vocabulary.
          target_vocab_size:    size of the target vocabulary.
          buckets:              a list of pairs (I, O), where I (O) specifies maximum input (output) length
                                that will be processed in that bucket
          size:                 number of units in each layer of the model.
          num_layers:           number of layers in the model.
          max_gradient_norm:    gradients will be clipped to maximally this norm.
          batch_size:           the size of the batches used during training;
          learning_rate:        learning rate to start with.
          lr_decay:             decay learning rate by this much when needed.
          num_samples:          number of samples for sampled softmax.
          forward_only:         if set, we do not construct the backward pass in the model.
          dtype:                the data type to use to store internal variables.
        """

        from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq, model_with_buckets

        # =====================================================================================================
        # Store instance variables.
        # =====================================================================================================

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets        = buckets
        self.batch_size     = batch_size
        self.learning_rate  = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.lr_decay_op    = self.learning_rate.assign(lr_decay * self.learning_rate)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)

        # =====================================================================================================
        # Define basic components: cell(s) state, encoder, decoder.
        # =====================================================================================================

        # If we use sampled softmax, we need an output projection.
        softmax_loss_function, output_projection = Seq2SeqModel._get_loss_fn(num_samples, size, self.target_vocab_size)

        # Create the internal (potentially multi-layer) cell for our RNN.
        def single_cell():
            return tf.contrib.rnn.GRUCell(size)
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
        else:
            cell = single_cell()

        # Feeds for encoder inputs.
        max_input_length    = buckets[-1][0]
        self.encoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)) for i in range(max_input_length)]

        # Feeds & weights for decoder inputs.
        max_output_length   = buckets[-1][1] + 1  # because we always append EOS after
        self.decoder_inputs = [tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)) for i in range(max_output_length)]
        self.target_weights = [tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)) for i in range(max_output_length)]

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]

        # =====================================================================================================
        # Combine the components to construct desired model architecture.
        # =====================================================================================================

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                               num_encoder_symbols=source_vocab_size, num_decoder_symbols=target_vocab_size,
                                               embedding_size=size, output_projection=output_projection,
                                               feed_previous=do_decode, dtype=dtype)

        # Note: if softmax_loss_func is None, model_with_buckets will default to standard softmax.
        self.outputs, self.losses = model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                                       targets, self.target_weights,
                                                       buckets, lambda x, y: seq2seq_f(x, y, forward_only),
                                                       softmax_loss_function=softmax_loss_function)

        # If we use output projection, we need to project outputs for decoding.
        if forward_only and output_projection is not None:
            self.outputs = Seq2SeqModel._get_projections(len(buckets), self.outputs, output_projection)

        # =====================================================================================================
        # Configure training process (backward pass).
        # =====================================================================================================

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            print("Looping over", len(buckets), "buckets.")
            for b in range(len(buckets)):
                print("\rCurrent bucket:", b, end="")
                sys.stdout.flush()
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        print("Creating saver and exiting . . . ")
        self.saver = tf.train.Saver(tf.global_variables())


    @staticmethod
    def _get_loss_fn(num_samples: int, hidden_size: int, target_vocab_size: int):
        """TODO
        :param num_samples:     (context: importance sampling) size of subset of outputs for softmax.
        :param hidden_size:     number of units in the individual recurrent states.
        :param target_vocab_size: number of unique output words.
        :return: sampled_loss, output_projection
        """
        should_sample = num_samples > 0 and num_samples < target_vocab_size
        if not should_sample: return None, None

        # Define the standard affine-softmax transformation from hidden_size -> vocab_size.
        # True output (for a given bucket) := tf.matmul(decoder_out, w) + b
        w_t = tf.get_variable("proj_w", [target_vocab_size, hidden_size], dtype=tf.float32)
        w = tf.transpose(w_t)
        b = tf.get_variable("proj_b", [target_vocab_size], dtype=tf.float32)
        output_projection = (w, b)

        def sampled_loss(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(
                    weights=w_t,
                    biases=b,
                    labels=labels,
                    inputs=inputs,
                    num_sampled=num_samples,
                    num_classes=target_vocab_size)

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
            projected_vals[b] = [tf.matmul(output, projection_operator[0]) + projection_operator[1] for output in unprojected_vals[b]]
        return projected_vals


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
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
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

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

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

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
