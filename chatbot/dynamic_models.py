"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pdb
from tqdm import tqdm
import time
import logging
import numpy as np
import tensorflow as tf
from chatbot._models import Model
from chatbot.model_components import *
from utils import io_utils
from utils.io_utils import GO_ID


class DynamicBot(Model):

    def __init__(self,
                 dataset,
                 batch_size=64,
                 ckpt_dir="out",
                 dropout_prob=0.0,
                 embed_size=32,
                 learning_rate=0.5,
                 lr_decay=0.99,
                 num_layers=2,
                 state_size=128,
                 steps_per_ckpt=200,
                 temperature=0.0,
                 is_chatting=False):
        """
        Args:
            dataset: 'Dataset' instance. Will likely be removed soon since it's only used
                      for grabbing quantities like vocab size.
            ckpt_dir: location where training checkpoint files will be saved.
            batch_size: number of samples per training step.
            state_size: number of nodes in the underlying RNN cell state.
            embed_size: size of embedding dimension that integer IDs get mapped into.
            learning_rate: float, typically in range [0, 1].
            lr_decay: weight decay factor, not strictly necessary since default optimizer is adagrad.
            steps_per_ckpt: (int) Specifies step interval for testing on validation data.
            dropout_prob: (float) in range [0., 1.]. probability of inputs being dropped,
                            applied before each layer in the model.
            num_layers: in the underlying MultiRNNCell. Total layers in model, not counting
                        recurrence/loop unrolliwng is then 2 * num_layers (encoder + decoder).
            temperature: determines how varied the bot responses will be when chatting.
                         The default (0.0) just results in deterministic argmax.
            is_chatting: boolean, should be False when training and True when chatting.
        """

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')

        self.state_size  = state_size
        self.embed_size  = embed_size
        self.vocab_size  = dataset.vocab_size
        # FIXME: Not sure how I feel about dataset as instance attribute.
        # It's helpful in the decoding/chat session, but shouldn't be an arg there either.
        self.dataset = dataset
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.batch_size = batch_size

        with tf.variable_scope("input_pipeline"):
            self.pipeline = InputPipeline(dataset.paths, batch_size, is_chatting=is_chatting)
            self.encoder_inputs = self.pipeline.encoder_inputs
            self.decoder_inputs = self.pipeline.decoder_inputs

        self.embedder = Embedder(self.vocab_size, embed_size)
        with tf.variable_scope("encoder") as encoder_scope:
            self.encoder = Encoder(state_size, self.embed_size,
                                   dropout_prob=dropout_prob,
                                   num_layers=num_layers)
            embedded_enc_inputs = self.embedder(self.encoder_inputs, scope=encoder_scope)
            # Get encoder state after feeding the sequence(s). Shape is [None, state_size].
            encoder_state = self.encoder(embedded_enc_inputs, scope=encoder_scope)

        with tf.variable_scope("decoder") as decoder_scope:
            self.decoder = Decoder(state_size, self.vocab_size, self.embed_size,
                                   dropout_prob=dropout_prob,
                                   num_layers=num_layers,
                                   temperature=temperature)
            # Decoder inputs in embedding space. Shape is [None, None, embed_size].
            embedded_dec_inputs = self.embedder(self.decoder_inputs, scope=decoder_scope)
            # For decoder, we want the full sequence of output states, not simply the last.
            decoder_outputs, decoder_state = self.decoder(embedded_dec_inputs,
                                                          initial_state=encoder_state,
                                                          is_chatting=is_chatting,
                                                          loop_embedder=self.embedder,
                                                          scope=decoder_scope)

        tf.summary.histogram("encoder_embedding", self.embedder.get_embed_tensor(encoder_scope))
        tf.summary.histogram("decoder_embedding", self.embedder.get_embed_tensor(decoder_scope))
        self.merged = tf.summary.merge_all()
        self.encoder_scope = encoder_scope
        self.decoder_scope = decoder_scope

        # If in chat session, need _projection from state space to vocab space.
        # Note: The decoder handles this for us (as it should).
        self.outputs = decoder_outputs

        # Let superclass handle the boring stuff (dirs/more instance variables).
        super(DynamicBot, self).__init__(self.log,
                                         dataset.name,
                                         ckpt_dir,
                                         dataset.vocab_size,
                                         batch_size,
                                         learning_rate,
                                         lr_decay,
                                         steps_per_ckpt,
                                         is_chatting)

    def compile(self, optimizer=None, max_gradient=5.0, reset=False):
        """ Configure training process and initialize model. Inspired by Keras.

        Args:
            optimizer: object that inherits from tf.train.Optimizer.
            max_gradient: float. Gradients will be clipped to be below this value.
            reset: boolean. Tells Model superclass whether or not we wish to compile
                            a model from scratch or load existing parameters from ckpt_dir.
        """

        if not self.is_chatting:
            with tf.name_scope("evaluation"):
                # Loss - target is to predict, as output, the next decoder input.
                # target_labels has shape [batch_size, dec_inp_seq_len - 1]
                target_labels = self.decoder_inputs[:, 1:]
                target_weights = tf.cast(self.decoder_inputs > 0, self.decoder_inputs.dtype)
                #self.loss = tf.losses.sparse_softmax_cross_entropy(
                #    labels=target_labels, logits=self.outputs[:, :-1, :],
                #    weights=target_weights[:, 1:])
                self.loss = self.sampled_softmax_loss(num_sampled=80)

                # Define the training portion of the graph.
                if optimizer is None:
                    optimizer = tf.train.AdagradOptimizer(self.learning_rate)
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient)
                grads = list(zip(clipped_gradients, params))
                self.apply_gradients = optimizer.apply_gradients(grads, global_step=self.global_step)
                correct_pred = tf.equal(
                    tf.argmax(self.outputs[:, :-1, :], axis=2), tf.argmax(target_labels)
                )
                accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
                accuracy /= tf.cast(tf.size(correct_pred), tf.float32)

            # Creating a summar.scalar tells TF that we want to track the value for visualization.
            # We can view plots of how they change over training in TensorBoard.
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('learning_rate', self.learning_rate),
            self.merged = tf.summary.merge_all()

        # Next, let superclass load param values from file (if not reset), otherwise
        # initialize newly created model.
        super(DynamicBot, self).compile(reset=reset)

    def step(self, forward_only=False):
        """Run forward and backward pass on single data batch.

        Args:
            forward_only: if True, don't perform backward pass (gradient updates).

        Returns:
            summaries, step_loss, step_outputs.
            If forward_only == False, then outputs is None
        """

        if not forward_only:
            fetches = [self.merged, self.loss, self.apply_gradients]
            summaries,step_loss, _ = self.sess.run(fetches)
            return summaries, step_loss, None
        if self.is_chatting:
            step_outputs = self.sess.run(self.outputs,
                                         feed_dict=self.pipeline.feed_dict)
            return None, None, step_outputs
        else:
            fetches = [self.merged, self.loss, self.outputs]
            summaries, step_loss, step_outputs = self.sess.run(fetches)
            return summaries, step_loss, step_outputs

    def train(self, dataset, nb_epoch=1):
        """Train bot on inputs for nb_epoch epochs, or until user types CTRL-C.

        Args:
            dataset: any instance of the Dataset class.
            nb_epoch: (int) Number of times to train over all entries in inputs.
        """

        def perplexity(loss):
            """Common alternative to loss in NLP models."""
            return np.exp(float(loss)) if loss < 300 else float("inf")

        self.embedder.assign_visualizer(self.file_writer,
                                        self.encoder_scope,
                                        dataset.paths['from_vocab'])
        self.embedder.assign_visualizer(self.file_writer,
                                        self.decoder_scope,
                                        dataset.paths['to_vocab'])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        # This is needed because apparently TensorFlow's coordinator isn't all that
        # great at, well, coordinating. If this makes you sad, it also makes me sad.
        # Feel free to email me for emotional support.
        print('QUEUE RUNNERS RELEASED.'); time.sleep(1)
        print('CAN THEY ENQUEUE IN TIME?'); time.sleep(2)
        print('HERE'); time.sleep(1)
        print('WE'); time.sleep(1)
        print('GO!!!!!')

        try:
            i_step = 0
            avg_loss = avg_step_time = 0.0
            while not coord.should_stop():

                start_time = time.time()
                summaries, step_loss, _ = self.step()
                # Calculate running averages.
                avg_step_time  += (time.time() - start_time) / self.steps_per_ckpt
                avg_loss       += step_loss / self.steps_per_ckpt

                # Print updates in desired intervals (steps_per_ckpt).
                if i_step % self.steps_per_ckpt == 0:
                    # Display averged-training updates and save.
                    print("Step %d:" % i_step, end=" ")
                    print("step time = %.3f" % avg_step_time)
                    print("\ttraining loss = %.3f" % avg_loss, end="; ")
                    print("training perplexity = %.2f" % perplexity(avg_loss))
                    self.save(summaries=summaries)

                    # Toggle data switch and led the validation flow!
                    self.pipeline.toggle_active()
                    summaries, eval_loss, _ = self.step(forward_only=True)
                    self.pipeline.toggle_active()
                    print("\tValidation loss = %.3f" % eval_loss, end="; ")
                    print("val perplexity = %.2f" % perplexity(eval_loss))
                    # Reset the running averages and exit checkpoint.
                    avg_loss = avg_step_time = 0.0

                i_step += 1
        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            coord.request_stop()
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError. You have run out of data. Get some more.")
            coord.request_stop()
        finally:
            coord.join(threads)
            self.close()

    def decode(self):
        """
        The higher the temperature, the more varied will be the bot's responses.
        """
        # We decode one sentence at a time.
        self.batch_size = 1
        assert self.is_chatting
        # Decode from standard input.
        print("Type \"exit\" to exit.")
        print("Write stuff after the \">\" below and I, your robot friend, will respond.")
        sentence = io_utils.get_sentence()
        while sentence:
            response = self(sentence)
            print(response)
            sentence = io_utils.get_sentence()
            if sentence == 'exit':
                print("Farewell, human.")
                break

    def __call__(self, sentence):
        """This is how we talk to the bot."""
        # Convert input sentence to token-ids.
        encoder_inputs = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(sentence),self.dataset.word_to_idx)

        encoder_inputs = np.array([encoder_inputs[::-1]])
        self.pipeline.feed_user_input(encoder_inputs)
        # Get output sentence from the chatbot.
        a, b, response = self.step(forward_only=True)
        assert a is None and b is None
        # response has shape [1, response_length] and it's last elemeot is EOS_ID. :)
        return self.dataset.as_words(response[0][:-1])


    def sampled_softmax_loss(self, num_sampled):
            """
            Args:
                output_projection: (tuple) returned by any Decoder.get_projections_tensors()
                    - output_projection[0] == w tensor. [state_size, vocab_size]
                    - output_projection[0] == b tensor. [vocab_size]
                labels: 2D Integer tensor. [batch_size, None]
                state_outputs: 3D float Tensor [batch_size, None, state_size].
                    - In this project, usually is the decoder batch output sequence (NOT projected).
                num_sampled: number of classes out of vocab_size possible to use.
                vocab_size: total number of classes.
            """

            def _sparse_to_dense(sampled_logits, labels, sampled, num_sampled):
                acc_hits = tf.nn.compute_accidental_hits(labels, sampled, num_true=1)
                acc_indices, acc_ids, acc_weights = acc_hits
                # This is how SparseToDense expects the indices.
                acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
                acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
                sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1, "sparse_indices")
                # Create sampled_logits_shape = [batch_size, num_sampled]
                sampled_logits_shape = tf.concat([tf.shape(labels)[:1], tf.expand_dims(num_sampled, 0)], 0)
                if sampled_logits.dtype != acc_weights.dtype:
                    acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
                return tf.sparse_to_dense(sparse_indices, sampled_logits_shape, acc_weights,
                                          default_value=0.0,validate_indices=False)

            # Extract transpose weights, now shape is [vocab_size, state_size].
            # Use tf.reshape which is dynamic as opposed to static (i.e. slow) tf.transpose.
            vocab_size = self.vocab_size
            state_outputs = self.outputs[:, :-1, :]
            labels = self.decoder_inputs[:, 1:]
            output_projection = self.decoder.get_projection_tensors()
            weights = tf.reshape(output_projection[0], [vocab_size, -1])
            state_size = tf.shape(weights)[-1]
            biases  = output_projection[1]

            with tf.name_scope("compute_sampled_logits", [weights, biases, state_outputs, labels]):
                labels = tf.cast(labels, tf.int64)
                # Smush tensors so we can use them with tensorflow methods.
                # Question: Docs suggest we should reshape to [-1, 1] so I'm keeping.
                # but original code had it as just [-1].
                labels_flat = tf.reshape(labels, [-1, 1])
                # Sample the negative labels. Returns 3-tuple:
                #   1. sampled_candidates: [num_sampled] tensor
                #   2. true_expected_count shape = [batch_size*None, 1] tensor
                #   ---- Entries associated 1-to-1 with smushed labels.
                #   3. sampled_expected_count shape = [num_sampled] tensor
                #   ---- Entries associated 1-to-1 with sampled_candidates.
                sampled_values = tf.nn.log_uniform_candidate_sampler(
                    true_classes=labels_flat, num_true=1, num_sampled=num_sampled,
                    unique=True, range_max=vocab_size)
                sampled, Q_true, Q_samp = (tf.stop_gradient(s) for s in sampled_values)
                sampled = tf.cast(sampled, tf.int64)

                # Casting this back to actually be flat.
                batch_times_none = tf.shape(labels_flat)[0]
                labels_flat = tf.reshape(labels, [-1])
                # Get concatenated 1D tensor of shape [batch_size * None + num_samples],
                all_ids = tf.concat([labels_flat, sampled], 0)

                # The embedding_lookup here should be thought of as embedding
                # the integer label and sampled IDs in the state space.
                # all_w has shape [batch_size * None + num_samples, state_size]
                # all_b has shape [batch_size * None + num_samples]
                with tf.control_dependencies([tf.assert_equal(state_size, self.state_size)]):
                    all_w       = tf.nn.embedding_lookup(weights, all_ids, partition_strategy='div')
                    all_b       = tf.nn.embedding_lookup(biases, all_ids)
                    true_w      = tf.slice(all_w, begin=[0, 0], size=[batch_times_none, state_size])
                    true_b      = tf.slice(all_b, begin=[0], size=[batch_times_none])
                    sampled_w   = tf.slice(all_w, begin=[batch_times_none, 0], size=[num_sampled, state_size])
                    sampled_b   = tf.slice(all_b, begin=[batch_times_none], size=[num_sampled])

                    state_outputs    = tf.reshape(state_outputs, [batch_times_none, state_size])
                    state_outputs = tf.cast(state_outputs, tf.float32)
                    true_logits      = tf.reduce_sum(tf.multiply(state_outputs, true_w), 1)
                true_logits     += true_b - tf.log(Q_true)
                # Matmul shapes [batch_times_none, state_size] * [state_size, num_sampled].
                sampled_logits   = tf.matmul(state_outputs, sampled_w, transpose_b=True) + sampled_b
                sampled_logits  += _sparse_to_dense(sampled_logits, tf.expand_dims(labels_flat, -1), sampled, num_sampled) - tf.log(Q_samp)
                sampled_logits  -= tf.log(Q_samp)

                # Construct output logits and labels. The true labels/logits start at col 0.
                # shape(out_logits) == [batch_times_none, 1 + num_sampled]. I'M SURE.
                out_logits = tf.concat([true_logits, sampled_logits], 1)
                # true_logits is a float tensor, ones_like(true_logits) is a float tensor of ones.
                # Question: wtf??
                out_labels = tf.concat([tf.ones_like(true_logits), tf.zeros_like(sampled_logits)], 1)

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=out_labels, logits=out_logits))
