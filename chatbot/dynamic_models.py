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
                 max_seq_len=None,
                 is_decoding=False):

        if max_seq_len is None:
            max_seq_len = dataset.max_seq_len

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('DynamicBotLogger')
        self.dataset     = dataset
        self.state_size  = state_size
        self.embed_size  = embed_size
        self.max_seq_len = max_seq_len

        # Thanks to variable scoping, only need one object for multiple embeddings/rnns.
        embedder    = Embedder(dataset.vocab_size, embed_size)
        dynamic_rnn = DynamicRNN(state_size)

        # ==========================================================================================
        # Input sentences are embedded and fed to an encoder.
        # ==========================================================================================

        #self.encoder_inputs = tf.placeholder(tf.int32, (None, max_seq_len))
        # If shape is not specified, you can feed any shape (what??)
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])

        # Create the embedder, then apply it on the inputs.
        embedded_enc_inputs = embedder(self.encoder_inputs, scope="encoder")
        # Create the encoder, then feed it the embedded inputs.
        encoder_state = dynamic_rnn(embedded_enc_inputs, scope="encoder")


        # ==========================================================================================
        # When training, decoder is fed embedded target response sentences.
        # ==========================================================================================

        #self.decoder_inputs = tf.placeholder(tf.int32, (None, max_seq_len + 1))
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
        # Create the embedder, then apply it on the inputs.
        embedded_dec_inputs = embedder(self.decoder_inputs, scope="decoder")
        # Create the decoder, then feed it the embedded inputs.
        decoder_outputs, decoder_state = dynamic_rnn(embedded_dec_inputs,
                                                 scope="decoder",
                                                 initial_state=encoder_state,
                                                 return_sequence=True)
        # Projection to vocab space.
        output_projection = OutputProjection(state_size, dataset.vocab_size)
        self.outputs = output_projection(decoder_outputs)
        #check_shape(self.outputs, [None, max_seq_len+1, dataset.vocab_size], self.log)
        check_shape(self.outputs, [None, None, dataset.vocab_size], self.log)

        # ==========================================================================================
        # Training/evaluation operations.
        # ==========================================================================================

        # Loss - target is to predict, as output, the next decoder input.
        target_labels = self.decoder_inputs[:, 1:]
        self.target_weights = tf.placeholder(tf.float32, [None, None])
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
        """ Configure training process and initialize model. Inspired by Keras."""

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
            encoder_inputs: shape [batch_size, max_time]
            decoder_inputs: shape [batch_size, max_time]

        Returns:
            self.is_decoding is True:
                loss: (scalar) for this batch.
            outputs: array with shape [batch_size, max_time+1, vocab_size]
        """

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

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs.name] = decoder_inputs
        input_feed[self.target_weights.name] = target_weights

        if not forward_only:
            fetches = [self.loss, self.apply_gradients]
            outputs = self.sess.run(fetches, input_feed)
            return outputs[0]  # loss
        else:
            fetches = [self.loss, self.outputs]
            outputs = self.sess.run(fetches, input_feed)
            return outputs[0], outputs[1]  # loss, outputs

    def train(self, encoder_inputs, decoder_inputs,
              nb_epoch=1, steps_per_ckpt=100, save_dir=None):
        i_step = 0
        avg_loss = 0.0
        avg_step_time = 0.0
        try:
            while True:
                start_time      = time.time()
                step_loss       = self.step(encoder_inputs[i_step], encoder_inputs[i_step])

                # Calculate running averages.
                avg_step_time  += (time.time() - start_time) / steps_per_ckpt
                avg_loss       += step_loss / steps_per_ckpt

                # Print updates in desired intervals (steps_per_ckpt).
                if i_step % steps_per_ckpt == 0:
                    self.save(save_dir)
                    print("Step {}: step time = {};  loss = {}".format(
                        i_step, avg_step_time, avg_loss))
                    # Reset the running averages.
                    avg_step_time = 0.0
                    avg_loss = 0.0
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
        assert(len(encoder_inputs.shape) == 2)
        # Get output sentence from the chatbot.
        _, logits = self.step(encoder_inputs, forward_only=True)

        # TODO: temperature sampling support soon.
        output_tokens  = logits[0].argmax(axis=1)
        # If there is an EOS symbol in outputs, cut them at that point.
        if io_utils.EOS_ID in output_tokens:
            output_tokens = output_tokens[:output_tokens.index(io_utils.EOS_ID)]

        return self.dataset.as_words(output_tokens)
