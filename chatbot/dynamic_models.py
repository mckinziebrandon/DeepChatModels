"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import tensorflow as tf
from utils import io_utils
from utils.io_utils import GO_ID
from chatbot._models import Model
from chatbot.model_components import *


def check_shape(tensor, expected_shape, log):
    if tensor.shape.as_list() != expected_shape:
        msg = "Bad shape of tensor {0}. Expected {1} but found {2}.".format(
            tensor.name, expected_shape, tensor.shape.as_list())
        log.error(msg)
        raise ValueError(msg)


class DynamicBot(Model):

    def __init__(self,
                 dataset,
                 ckpt_dir="out",
                 batch_size=64,
                 state_size=256,
                 embed_size=32,
                 learning_rate=0.4,
                 lr_decay=0.98,
                 max_seq_len=500,
                 is_decoding=False):
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
            max_seq_len: maximum allowed number of words per sentence.
            is_decoding: boolean, should be False when training and True when chatting.
        """

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')
        self.state_size  = state_size
        self.embed_size  = embed_size
        self.max_seq_len = max_seq_len
        self.vocab_size  = dataset.vocab_size
        # FIXME: Not sure how I feel about dataset as instance attribute.
        # It's quite helpful in the decoding/chat sessions, but it feels even more odd
        # passing it as an argument there.
        self.dataset = dataset

        # Thanks to variable scoping, only need one object for multiple embeddings/rnns.
        embedder    = Embedder(self.vocab_size, embed_size)
        self.dynamic_rnn = DynamicRNN(state_size, self.vocab_size)

        # The following placeholder shapes correspond with [batch_size, seq_len].
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.target_weights = tf.placeholder(tf.float32, [None, None])

        # Encoder inputs in embedding space. Shape is [None, None, embed_size].
        embedded_enc_inputs = embedder(self.encoder_inputs, scope="encoder")
        # Get encoder state after feeding the sequence(s). Shape is [None, state_size].
        encoder_state = self.dynamic_rnn(embedded_enc_inputs, scope="encoder")

        # Decoder inputs in embedding space. Shape is [None, None, embed_size].
        embedded_dec_inputs = embedder(self.decoder_inputs, scope="decoder")
        # For decoder, we want the full sequence of output states, not simply the last.
        decoder_outputs, decoder_state = self.dynamic_rnn(
            embedded_dec_inputs,
            initial_state=encoder_state,
            return_sequence=True,
            is_decoding=is_decoding,scope="decoder")

        # Projection from state space to vocab space.
        self.outputs = decoder_outputs

        if not is_decoding:
            check_shape(self.outputs, [None, None, dataset.vocab_size], self.log)
            # Loss - target is to predict, as output, the next decoder input.
            target_labels = self.decoder_inputs[:, 1:]
            check_shape(target_labels, [None, None], self.log)
            self.loss = tf.losses.sparse_softmax_cross_entropy(
                labels=target_labels, logits=self.outputs[:, :-1, :], weights=self.target_weights
            )

        # Let superclass handle the boring stuff (dirs/more instance variables).
        super(DynamicBot, self).__init__(dataset.name,
                                         ckpt_dir,
                                         dataset.vocab_size,
                                         batch_size,
                                         learning_rate,
                                         lr_decay,
                                         is_decoding)

    def compile(self, optimizer=None, max_gradient=5.0, reset=False):
        """ Configure training process and initialize model. Inspired by Keras.

        Args:
            optimizer: instance of tf.train.Optimizer. Defaults to AdagradOptimizer.
            max_gradient: float. Gradients will be clipped to be below this value.
            reset: boolean. Tells Model superclass whether or not we wish to compile
                            a model from scratch or load existing parameters from ckpt_dir.
        """

        if not self.is_decoding:
            # First, define the training portion of the graph.
            params = tf.trainable_variables()
            if optimizer is None:
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 10.0)
            self.apply_gradients = optimizer.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        # Next, let superclass load param values from file (if not reset), otherwise
        # initialize newly created model.
        super(DynamicBot, self).compile(reset=reset)

    def step(self, encoder_inputs, decoder_inputs=None, forward_only=False):
        """Run forward and backward pass on single data batch.

        Args:
            encoder_inputs: token ids with shape [batch_size, max_time].
            decoder_inputs: None, or token ids with shape [batch_size, max_time].
            forward_only: if True, don't perform backward pass (gradient updates).

        Returns:
            step_loss, step_outputs. If forward_only == False, then outputs is None

        """

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs

        if forward_only and decoder_inputs is None:
            decoder_inputs = np.array([[GO_ID]])
            target_weights = np.array([[1.0]])
        else:
            decoder_inputs = [np.hstack(([GO_ID], sent)) for sent in decoder_inputs]
            target_weights = list(np.ones(shape=(self.batch_size, self.max_seq_len)))
            for b in range(self.batch_size):
                for m in range(self.max_seq_len):
                    if decoder_inputs[b][m+1] == io_utils.PAD_ID:
                        target_weights[b][m] = 0.0
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.target_weights.name] = target_weights


        if not forward_only:
            fetches = [self.loss, self.apply_gradients]
            step_loss, _ = self.sess.run(fetches, input_feed)
            return step_loss, None
        else:
            if self.is_decoding:
                fetches = [self.outputs]
                step_outputs = self.sess.run(fetches, input_feed)
                return None, step_outputs[0]
            else:
                fetches = [self.loss, self.outputs]
                step_loss, step_outputs = self.sess.run(fetches, input_feed)
                return step_loss, step_outputs

    def train(self, train_data, valid_data,
              nb_epoch=1, steps_per_ckpt=100, save_dir=None):
        """Train bot on inputs for nb_epoch epochs, or until user types CTRL-C.

        Args:
            train_data:
            valid_data:
            nb_epoch: (int) Number of times to train over all entries in inputs.
            steps_per_ckpt: (int) Specifies step interval for testing on validation data.
            save_dir: (str) Path to save ckpt files. If None, defaults to self.ckpt_dir.
        """

        print("Preparing data batches . . . ")

        # Get training data in proper format.
        encoder_inputs_train, decoder_inputs_train = io_utils.batch_concatenate(
            train_data, self.batch_size, self.max_seq_len)

        # Get validation data in proper format.
        encoder_inputs_valid, decoder_inputs_valid = io_utils.batch_concatenate(
            valid_data, self.batch_size, self.max_seq_len)

        i_step = 0
        avg_loss = avg_step_time = 0.0
        try:
            while True:
                start_time = time.time()
                step_loss, _ = self.step(encoder_inputs_train[i_step],
                                         encoder_inputs_train[i_step])

                # Calculate running averages.
                avg_step_time  += (time.time() - start_time) / steps_per_ckpt
                avg_loss       += step_loss / steps_per_ckpt

                # Print updates in desired intervals (steps_per_ckpt).
                if i_step % steps_per_ckpt == 0:
                    self.save(save_dir)
                    print("Step %d: step time = %.3f;  train loss = %.3f"
                          % (i_step, avg_step_time, avg_loss))

                    eval_loss, _ = self.step(encoder_inputs_train[i_step],
                                             encoder_inputs_train[i_step])
                    eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                    print("\tEval: loss = %.3f;  perplexity = %.3f" % (eval_loss, eval_ppx))
                    # Reset the running averages.
                    avg_loss = avg_step_time = 0.0
                i_step += 1

        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            self.save(save_dir)

    def decode(self):
        # We decode one sentence at a time.
        self.batch_size = 1
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

        encoder_inputs = np.array([encoder_inputs])
        # Get output sentence from the chatbot.
        _, logits = self.step(encoder_inputs, forward_only=True)

        output_tokens  = np.array(logits).argmax(axis=2).flatten()
        # If there is an EOS symbol in outputs, cut them at that point.
        if io_utils.EOS_ID in output_tokens:
            output_tokens = output_tokens[:output_tokens.index(io_utils.EOS_ID)]

        return self.dataset.as_words(output_tokens)
