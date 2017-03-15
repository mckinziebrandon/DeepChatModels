"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time
import logging
import numpy as np
import tensorflow as tf
from chatbot._models import Model
from chatbot.model_components import *
from utils import io_utils
from utils.io_utils import GO_ID
from heapq import *


def check_shape(tensor, expected_shape, log):
    if tensor.shape.as_list() != expected_shape:
        msg = "Bad shape of tensor {0}. Expected {1} but found {2}.".format(
            tensor.name, expected_shape, tensor.shape.as_list())
        log.error(msg)
        raise ValueError(msg)


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
                 steps_per_ckpt=100,
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
        # It's quite helpful in the decoding/chat sessions, but it feels even more odd
        # passing it as an argument there.
        self.dataset = dataset
        self.init_learning_rate = learning_rate
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        # The following placeholder shapes correspond with [batch_size, seq_len].
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.target_weights = tf.placeholder(tf.float32, [None, None], name="target_weights")

        # Thanks to variable scoping, only need one object for multiple embeddings.
        self.embedder = Embedder(self.vocab_size, embed_size)


        # Encoder inputs in embedding space. Shape is [None, None, embed_size].
        with tf.variable_scope("encoder") as encoder_scope:
            self.encoder = Encoder(state_size, self.embed_size,
                                   dropout_prob=dropout_prob, num_layers=num_layers)
            embedded_enc_inputs = self.embedder(self.encoder_inputs, scope=encoder_scope)
            # Get encoder state after feeding the sequence(s). Shape is [None, state_size].
            encoder_state = self.encoder(embedded_enc_inputs, scope=encoder_scope)

        with tf.variable_scope("decoder") as decoder_scope:
            self.decoder = Decoder(state_size, self.vocab_size, self.embed_size,
                                   dropout_prob=dropout_prob, num_layers=num_layers,
                                   temperature=temperature)
            # Decoder inputs in embedding space. Shape is [None, None, embed_size].
            embedded_dec_inputs = self.embedder(self.decoder_inputs, scope=decoder_scope)
            # For decoder, we want the full sequence of output states, not simply the last.
            decoder_outputs, decoder_state = self.decoder(embedded_dec_inputs,
                                                          initial_state=encoder_state,
                                                          is_chatting=is_chatting,
                                                          loop_embedder=self.embedder,
                                                          scope=decoder_scope)

        #with tf.variable_scope("summaries") as self.summary_scope:
        #    self.summary_scope = self.summary_scope  # *sighs audibly*
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
            max_gradient: float. Gradients will be clipped to be below this value.
            reset: boolean. Tells Model superclass whether or not we wish to compile
                            a model from scratch or load existing parameters from ckpt_dir.
        """

        if not self.is_chatting:
            # Define ops/variables related to loss computation.
            with tf.variable_scope("evaluation"):
                check_shape(self.outputs, [None, None, self.vocab_size], self.log)
                # Loss - target is to predict, as output, the next decoder input.
                target_labels = self.decoder_inputs[:, 1:]
                check_shape(target_labels, [None, None], self.log)
                self.loss = tf.losses.sparse_softmax_cross_entropy(
                    labels=target_labels, logits=self.outputs[:, :-1, :],
                    weights=self.target_weights[:, :-1])

                # Define the training portion of the graph.
                params = tf.trainable_variables()
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient)
                grads = list(zip(clipped_gradients, params))
                self.apply_gradients = optimizer.apply_gradients(grads, global_step=self.global_step)

            # Creating a summar.scalar tells TF that we want to track the value for visualization.
            # It is the responsibility of the bot to save these via train_writer after each step.
            # We can view plots of how they change over training in TensorBoard.
            # with tf.variable_scope(self.summary_scope):
            tf.summary.scalar("loss", self.loss),
            tf.summary.scalar("learning_rate", self.learning_rate)
            self.merged = tf.summary.merge_all()

        # Next, let superclass load param values from file (if not reset), otherwise
        # initialize newly created model.
        super(DynamicBot, self).compile(reset=reset)

    def step(self, encoder_batch, decoder_batch=None, forward_only=False):
        """Run forward and backward pass on single data batch.

        Args:
            encoder_batch: integer numpy array with shape [batch_size, seq_len].
            decoder_batch: None, or numpy array with shape [batch_size, seq_len].
            forward_only: if True, don't perform backward pass (gradient updates).

        Returns:
            summaries, step_loss, step_outputs.
            If forward_only == False, then outputs is None
        """
        if decoder_batch is None and not forward_only:
            self.log.error("Can't perform gradient updates without a decoder_batch.")

        if self.is_chatting:
            assert decoder_batch is None, "Found decoder_batch inputs during chat session."
            decoder_batch = np.array([[GO_ID]])
            target_weights = np.array([[1.0]])
        else:
            # Prepend GO token to each sample in decoder_batch.
            decoder_batch   = np.insert(decoder_batch, 0, [GO_ID], axis=1)
            target_weights  = np.ones(shape=decoder_batch.shape)
            pad_indices = np.where(decoder_batch == io_utils.PAD_ID)
            # Define weights to be 0 if next decoder input is PAD.
            for batch_idx, time_idx in np.stack(pad_indices, axis=1):
                target_weights[batch_idx, time_idx-1] = 0.0
            # Last element should never be accessed anyway, since the target for a given
            # decoder input is defined as the next decoder input, but better to be safe.
            target_weights[:, -1] = 0.0

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_batch
        input_feed[self.decoder_inputs.name] = decoder_batch
        input_feed[self.target_weights.name] = target_weights

        if not forward_only:
            fetches = (self.merged, self.loss, self.apply_gradients)
            summaries, step_loss, _ = self.sess.run(fetches, input_feed)
            return summaries, step_loss, None
        elif self.is_chatting:
            step_outputs = self.sess.run(self.outputs, input_feed)
            return None, None, step_outputs
        else:
            fetches = [self.merged, self.loss, self.outputs]
            summaries, step_loss, step_outputs = self.sess.run(fetches, input_feed)
            return summaries, step_loss, step_outputs

    def train(self, dataset, nb_epoch=1, save_dir=None, searching_hyperparams=False):
        """Train bot on inputs for nb_epoch epochs, or until user types CTRL-C.

        Args:
            dataset: any instance of the Dataset class.
            nb_epoch: (int) Number of times to train over all entries in inputs.
            save_dir: (str) Path to save ckpt files. If None, defaults to self.ckpt_dir.
        """

        def perplexity(loss):
            """Common alternative to loss in NLP models."""
            return np.exp(float(loss)) if loss < 300 else float("inf")

        def save_loss_and_hyperparams(validation_loss, file_name=None):
            """Save snapshot of hyperparams with current validation loss."""
            if file_name is None:
                file_name = 'data/saved_train_data/' + self.data_name + '.csv'
            hyper_params = {"global_step":[self.global_step.eval(session=self.sess)],
                           "loss": [eval_loss],
                            "learning_rate":[self.init_learning_rate],
                            "vocab_size":[self.vocab_size],
                            "state_size":[self.state_size],
                            "embed_size":[self.embed_size],
                            "dropout_prob":[self.dropout_prob],
                            "num_layers":[self.num_layers]}
            io_utils.save_hyper_params(hyper_params, fname=file_name)

        self.embedder.assign_visualizer(self.train_writer, self.encoder_scope)
        try:
            for i_epoch in range(nb_epoch):

                i_step = 0
                avg_loss = avg_step_time = 0.0
                # Create data generators for feeding inputs to step().
                train_gen = dataset.train_generator(self.batch_size)
                valid_gen = dataset.valid_generator(self.batch_size)

                print("_______________ NEW EPOCH: %d _______________" % i_epoch)
                for encoder_batch, decoder_batch in train_gen:
                    start_time = time.time()
                    summaries, step_loss, _ = self.step(encoder_batch, decoder_batch)
                    # Calculate running averages.
                    avg_step_time  += (time.time() - start_time) / self.steps_per_ckpt
                    avg_loss       += step_loss / self.steps_per_ckpt

                    # Print updates in desired intervals (steps_per_ckpt).
                    if i_step % self.steps_per_ckpt == 0:

                        print("Step %d:" % i_step, end=" ")
                        print("step time = %.3f" % avg_step_time)
                        print("\ttraining loss = %.3f" % avg_loss, end="; ")
                        print("training perplexity = %.1f" % perplexity(avg_loss))
                        self.save(summaries=summaries, summaries_type="train", save_dir=save_dir)

                        # Run validation step. If we are out of validation data, reset generator.
                        try:
                            summaries, eval_loss, _ = self.step(*next(valid_gen))
                        except StopIteration:
                            valid_gen = dataset.valid_generator(self.batch_size)
                            summaries, eval_loss, _ = self.step(*next(valid_gen))

                        print("\tValidation loss = %.3f" % eval_loss, end="; ")
                        print("val perplexity = %.1f" % perplexity(eval_loss))
                        self.save(summaries=summaries, summaries_type="valid", save_dir=save_dir)

                        if searching_hyperparams and self.data_name != 'test_data':
                            save_loss_and_hyperparams(eval_loss)

                        # Reset the running averages and exit checkpoint.
                        avg_loss = avg_step_time = 0.0

                    i_step += 1

        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
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
        # Get output sentence from the chatbot.
        a, b, response = self.step(encoder_inputs, forward_only=True)
        assert a is None
        assert b is None
        # response has shape [1, response_length] and it's last element is EOS_ID. :)
        return self.dataset.as_words(response[0][:-1])
