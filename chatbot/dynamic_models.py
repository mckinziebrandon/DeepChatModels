"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import logging
import numpy as np
import tensorflow as tf
from chatbot._models import Model
from chatbot.model_components import *
from utils import io_utils
from utils.io_utils import GO_ID, batch_padded, batch_generator


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
                 state_size=128,
                 embed_size=32,
                 learning_rate=0.6,
                 lr_decay=0.995,
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

        # The following placeholder shapes correspond with [batch_size, seq_len].
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.target_weights = tf.placeholder(tf.float32, [None, None], name="target_weights")

        # Thanks to variable scoping, only need one object for multiple embeddings.
        embedder = Embedder(self.vocab_size, embed_size)
        #cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(self.state_size)
        #                                    for _ in range(self.num_layers)])

        # Encoder inputs in embedding space. Shape is [None, None, embed_size].
        with tf.variable_scope("encoder") as encoder_scope:
            self.encoder = Encoder(state_size, self.embed_size)
            embedded_enc_inputs = embedder(self.encoder_inputs, scope=encoder_scope)
            # Get encoder state after feeding the sequence(s). Shape is [None, state_size].
            encoder_state = self.encoder(embedded_enc_inputs, scope=encoder_scope)

        with tf.variable_scope("decoder") as decoder_scope:
            self.decoder = Decoder(state_size, self.vocab_size, self.embed_size, temperature)
            # Decoder inputs in embedding space. Shape is [None, None, embed_size].
            embedded_dec_inputs = embedder(self.decoder_inputs, scope=decoder_scope)
            # For decoder, we want the full sequence of output states, not simply the last.
            decoder_outputs, decoder_state = self.decoder(embedded_dec_inputs,
                                                          initial_state=encoder_state,
                                                          is_chatting=is_chatting,
                                                          loop_embedder=embedder,
                                                          scope=decoder_scope)

        with tf.variable_scope("summaries") as self.summary_scope:
            self.summary_scope = self.summary_scope  # *sighs audibly*
            tf.summary.histogram("embed_tensor", embedder.get_embed_tensor(encoder_scope))
            tf.summary.histogram("embed_tensor", embedder.get_embed_tensor(decoder_scope))
            self.merged = tf.summary.merge_all()

        # Projection from state space to vocab space.
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

                #_______  Sampled Softmax Construction status: stalled because ambiguity of
                # 'inputs' for sampled softmax documentation. What an odd implementation.
                #w, b = self.decoder.get_output_projection()
                #w_t = tf.transpose(w)
                #losses = []
                ##for i, label in enumerate(tf.unstack(target_labels, axis=1)):
                #for i in range(tf.shape(target_labels[0])):
                #    losses.append(tf.nn.sampled_softmax_loss(
                #        weights=w_t,
                #        biases=b,
                #        labels=target_labels[:, i],
                #        inputs=self.outputs[:, i],
                #        num_sampled=512,
                #        num_classes=self.vocab_size
                #    ))
                ## Welp, that should do it. Right?
                #self.loss = tf.stack(losses)


                # Define the training portion of the graph.
                params = tf.trainable_variables()
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
                gradients = tf.gradients(self.loss, params)
                clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient)
                grads = list(zip(clipped_gradients, params))
                self.apply_gradients = optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step)

            # Creating a summar.scalar tells TF that we want to track the value for visualization.
            # It is the responsibility of the bot to save these via train_writer after each step.
            # We can view plots of how they change over training in TensorBoard.
            with tf.variable_scope(self.summary_scope):
                tf.summary.scalar("loss", self.loss),
                tf.summary.scalar("learning_rate", self.learning_rate)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
                for grad, var in grads:
                    if grad is None: continue
                    tf.summary.histogram(var.name + "/gradient", grad)
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
            step_loss, step_outputs. If forward_only == False, then outputs is None

        """
        if decoder_batch is None and not forward_only:
            self.log.error("Can't perform gradient updates without a decoder_batch.")

        if forward_only and decoder_batch is None:
            # This is true if and only if we are in a chat session.
            # In a chat session, our batch size is 1 by definition, and
            # the inputs are fed 1 token at a time.
            decoder_batch = np.array([[GO_ID]])
            target_weights = np.array([[1.0]])
        else:
            # Prepend GO token to each sample in decoder_batch.
            decoder_batch   = np.insert(decoder_batch, 0, [GO_ID], axis=1)
            # Define weights to be 0 if next decoder input is PAD, else 1.
            target_weights  = np.ones(shape=decoder_batch.shape)
            pad_indices = np.where(decoder_batch == io_utils.PAD_ID)
            for b, m in np.stack(pad_indices, axis=1):
                target_weights[b, m-1] = 0.0
            # Last element should never be accessed anyway, but better be safe.
            target_weights[:, -1] = 0.0

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_batch
        input_feed[self.decoder_inputs.name] = decoder_batch
        input_feed[self.target_weights.name] = target_weights

        # TODO: Needs refactor.
        if not forward_only:
            fetches = (self.merged, self.loss, self.apply_gradients)
            summaries, step_loss, _ = self.sess.run(fetches, input_feed)
            return summaries, step_loss, None
        else:
            if self.is_chatting:
                step_outputs = self.sess.run(self.outputs, input_feed)
                return None, None, step_outputs
            else:
                fetches = [self.merged, self.loss, self.outputs]
                summaries, step_loss, step_outputs = self.sess.run(fetches, input_feed)
                return summaries, step_loss, step_outputs

    def train(self, train_data, valid_data,
              nb_epoch=1, save_dir=None):
        """Train bot on inputs for nb_epoch epochs, or until user types CTRL-C.

        Args:
            train_data: (2-tuple) property of any 'Dataset' instance.
            valid_data: (2-tuple) property of any 'Dataset' instance.
            nb_epoch: (int) Number of times to train over all entries in inputs.
            save_dir: (str) Path to save ckpt files. If None, defaults to self.ckpt_dir.
        """

        tf.logging.set_verbosity(tf.logging.ERROR)

        def perplexity(loss):
            """Common alternative to loss in NLP models."""
            return np.exp(float(loss)) if loss < 300 else float("inf")

        print("Preparing data batches . . . ")
        # Get training data as batch_padded lists.
        encoder_inputs_train, decoder_inputs_train = batch_padded(train_data, self.batch_size)
        # Get validation data as batch-padded lists.
        encoder_inputs_valid, decoder_inputs_valid = batch_padded(valid_data, self.batch_size)

        hyper_params = {}
        try:
            for i_epoch in range(nb_epoch):
                i_step = 0
                avg_loss = avg_step_time = 0.0
                # Create data generators.
                train_gen = batch_generator(encoder_inputs_train, decoder_inputs_train)
                valid_gen = batch_generator(encoder_inputs_valid, decoder_inputs_valid)
                print("_______________ NEW EPOCH: %d _______________" % i_epoch)
                for encoder_batch, decoder_batch in train_gen:
                    start_time = time.time()
                    summaries, step_loss, _ = self.step(encoder_batch, decoder_batch)
                    # Calculate running averages.
                    avg_step_time  += (time.time() - start_time) / self.steps_per_ckpt
                    avg_loss       += step_loss / self.steps_per_ckpt

                    # Print updates in desired intervals (steps_per_ckpt).
                    if i_step % self.steps_per_ckpt == 0:
                        # Save current parameter values in a new checkpoint file.
                        self.save(summaries=summaries, summaries_type="train", save_dir=save_dir)
                        # Report training averages.
                        print("Step %d: step time = %.3f;  perplexity = %.3f"
                              % (i_step, avg_step_time, perplexity(avg_loss)))
                        # Generate & run a batch of validation data.
                        summaries, eval_loss, _ = self.step(*next(valid_gen))
                        self.save(summaries=summaries, summaries_type="valid", save_dir=save_dir)
                        print("Validation loss:%.3f, perplexity:%.3f" % (eval_loss, perplexity(eval_loss)))

                        # TODO: less ugly. For now, having training up and running is priority.
                        if False:
                            hyper_params = {"global_step":[self.global_step.eval(session=self.sess)],
                                           "loss": [eval_loss],
                                            "learning_rate":[self.init_learning_rate],
                                            "vocab_size":[self.vocab_size],
                                            "state_size":[self.state_size],
                                            "embed_size":[self.embed_size]}
                            if i_step == 0:
                                hyper_params["loss"] = [step_loss]
                            if self.data_name != "test_data":
                                io_utils.save_hyper_params(
                                    hyper_params, fname='data/saved_train_data/'+self.data_name+".csv")

                        # Reset the running averages.
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
