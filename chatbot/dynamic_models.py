"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import tensorflow as tf
from chatbot import bot_ops
from chatbot._models import Model
from chatbot.recurrent_components import *
from chatbot.input_components import *
from utils import io_utils


class DynamicBot(Model):

    def __init__(self,
                 dataset,
                 batch_size=64,
                 ckpt_dir="out",
                 dropout_prob=0.0,
                 embed_size=None,
                 learning_rate=0.5,
                 lr_decay=0.99,
                 num_layers=2,
                 num_samples=512,
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
                        If None, will be set to state_size.
            learning_rate: float, typically in range [0, 1].
            lr_decay: weight decay factor, not strictly necessary since default optimizer is adagrad.
            steps_per_ckpt: (int) Specifies step interval for testing on validation data.
            dropout_prob: (float) in range [0., 1.]. probability of inputs being dropped,
                            applied before each layer in the model.
            num_layers: in the underlying MultiRNNCell. Total layers in model, not counting
                        recurrence/loop unrolliwng is then 2 * num_layers (encoder + decoder).
            num_samples: (int) size of subset of vocabulary_size to use for sampled softmax.
                         Require that 0 < num_samples < vocab size.
            temperature: determines how varied the bot responses will be when chatting.
                         The default (0.0) just results in deterministic argmax.
            is_chatting: boolean, should be False when training and True when chatting.
        """

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')

        if embed_size is None:
            embed_size = state_size
        # FIXME: Not sure how I feel about dataset as instance attribute.
        self.dataset        = dataset
        self.batch_size     = batch_size
        self.embed_size     = embed_size
        self.vocab_size     = dataset.vocab_size
        self.dropout_prob   = dropout_prob
        self.num_layers     = num_layers
        self.num_samples    = num_samples
        self.state_size     = state_size

        with tf.variable_scope("input_pipeline"):
            self.pipeline = InputPipeline(dataset.paths, batch_size, is_chatting=is_chatting)
            self.encoder_inputs = self.pipeline.encoder_inputs
            self.decoder_inputs = self.pipeline.decoder_inputs
            self.embedder = Embedder(self.vocab_size, embed_size)

        with tf.variable_scope("encoder") as encoder_scope:
            embedded_enc_inputs, embed_tensor = self.embedder(self.encoder_inputs)
            self.encoder = Encoder(state_size, self.embed_size,
                                   dropout_prob=dropout_prob,
                                   num_layers=num_layers,
                                   scope=encoder_scope)
            # Get encoder state after feeding the sequence(s). Shape is [None, state_size].
            encoder_state = self.encoder(embedded_enc_inputs)
            # Note: Histogram names are chosen for nice splitting in TensorBoard.
            tf.summary.histogram("embedding_encoder", embed_tensor)

        with tf.variable_scope("decoder") as decoder_scope:
            # Decoder inputs in embedding space. Shape is [None, None, embed_size].
            embedded_dec_inputs, embed_tensor = self.embedder(self.decoder_inputs)
            self.decoder = Decoder(state_size, self.vocab_size, self.embed_size,
                                   dropout_prob=dropout_prob,
                                   num_layers=num_layers,
                                   temperature=temperature,
                                   scope=decoder_scope)
            # For decoder, we want the full sequence of output states, not simply the last.
            decoder_outputs, decoder_state = self.decoder(embedded_dec_inputs,
                                                          initial_state=encoder_state,
                                                          is_chatting=is_chatting,
                                                          loop_embedder=self.embedder)
            tf.summary.histogram("embedding_decoder", embed_tensor)

        self.merged = tf.summary.merge_all()
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

    def compile(self, optimizer=None, max_gradient=5.0, reset=False, sampled_loss=True):
        """ Configure training process and initialize model. Inspired by Keras.

        Args:
            optimizer: object that inherits from tf.train.Optimizer.
            max_gradient: float. Gradients will be clipped to be below this value.
            reset: boolean. Tells Model superclass whether or not we wish to compile
                            a model from scratch or load existing parameters from ckpt_dir.
            sampled_loss: (bool) gives user the option to toggle sampled_loss
                          on/off post-initialization.
        """

        if not self.is_chatting:
            with tf.name_scope("evaluation") as scope:
                # Loss - target is to predict, as output, the next decoder input.
                # target_labels has shape [batch_size, dec_inp_seq_len - 1]
                target_labels = self.decoder_inputs[:, 1:]
                target_weights = tf.cast(self.decoder_inputs > 0, self.decoder_inputs.dtype)
                if sampled_loss and 0 < self.num_samples < self.vocab_size:
                    self.loss = bot_ops.dynamic_sampled_softmax_loss(
                        target_labels, self.outputs[:, :-1, :],
                        self.decoder.get_projection_tensors(), self.vocab_size)
                else:
                    self.loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=target_labels, logits=self.outputs[:, :-1, :],
                        weights=target_weights[:, 1:])

                # Define the training portion of the graph.
                if optimizer is None:
                    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
                params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient)
                grads = list(zip(clipped_gradients, params))
                self.apply_gradients = optimizer.apply_gradients(grads, global_step=self.global_step)

                # Computed accuracy, ensuring we use fully projected outputs.
                proj = self.outputs if not sampled_loss else self.decoder.apply_projection(self.outputs, scope)
                correct_pred = tf.equal(tf.round(tf.argmax(proj[:, :-1, :], axis=2)),
                                        tf.round(tf.argmax(target_labels)))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # Creating a summar.scalar tells TF that we want to track the value for visualization.
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('loss', self.loss),
            tf.summary.scalar('learning_rate', self.learning_rate),
            self.merged = tf.summary.merge_all()

        # Next, let superclass load param values from file (if not reset), otherwise
        # initialize newly created model.
        super(DynamicBot, self).compile(reset=reset)

    def step(self, forward_only=False):
        """Run one step of the model, which can mean 1 of the following:
            1. forward_only == False. This means we are training, so we should do both a
               forward and a backward pass.
            2. self.is_chatting. When chatting, we just get the response (word ID sequence).
            3. default to inference. Do a forward pass, but also compute loss(es) and summaries.

        Args:
            forward_only: if True, don't perform backward pass (gradient updates).

        Returns:
            summaries, step_loss, step_outputs.
            If forward_only == False, then outputs is None
        """

        if not forward_only:
            fetches = [self.merged, self.loss, self.apply_gradients]
            summaries, step_loss, _ = self.sess.run(fetches)
            return summaries, step_loss, None
        elif self.is_chatting:
            response = self.sess.run(self.outputs, feed_dict=self.pipeline.feed_dict)
            return None, None, response
        else:
            fetches = [self.merged, self.loss, self.outputs]
            summaries, step_loss, step_outputs = self.sess.run(fetches)
            return summaries, step_loss, step_outputs

    def train(self, dataset):
        """Train bot on inputs until user types CTRL-C or queues run out of data.

        Args:
            dataset: any instance of the Dataset class.
        """

        def perplexity(loss): return np.exp(float(loss)) if loss < 300 else float("inf")

        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        # Tell embedder to coordinate with TensorBoard's embedding visualization.
        # This allows to view, e.g., our words in 3D-projected embedding space (with labels!).
        label_paths = [dataset.paths['from_vocab'], dataset.paths['to_vocab']]
        self.embedder.assign_visualizer(self.file_writer, 'encoder', label_paths[0])
        self.embedder.assign_visualizer(self.file_writer, 'decoder', label_paths[1])

        # Note: Calling sleep(...) appears to allow sustained GPU utilization across training.
        # Without it, looks like GPU has to wait for data to be enqueued more often. Strange.
        print('QUEUE RUNNERS RELEASED.'); time.sleep(4)

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
        _, _, response = self.step(forward_only=True)
        # response has shape [1, response_length] and it's last elemeot is EOS_ID. :)
        return self.dataset.as_words(response[0][:-1])
