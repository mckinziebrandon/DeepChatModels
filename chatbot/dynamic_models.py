"""Sequence-to-sequence models with dynamic unrolling and faster embedding techniques."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import numpy as np
import tensorflow as tf
from chatbot._models import Model
from chatbot.components import bot_ops
from chatbot.components import InputPipeline, Embedder, BasicEncoder, BasicDecoder
from utils import io_utils
from pydoc import locate


class DynamicBot(Model):

    def __init__(self, dataset, params):
        """ General sequence-to-sequence model for conversations. Will eventually support
            attention, beam search, and a wider variety of cell options. At present,
            supports multi-layer encoder/decoders, GRU/LSTM cells, and fully dynamic
            unrolling (online decoding included). Additionally, will soon support
            biologically inspired mechanisms for learning, such as hebbian-based
            update rules.

        Args:
            dataset: any instance inheriting from data.DataSet.
            params: dictionary of hyperparameters.
                    See DEFAULT_FULL_CONFIG in chatbot._models.py for supported keys.
        """

        logging.basicConfig(level=logging.WARN)
        self.log = logging.getLogger('DynamicBotLogger')
        # Let superclass handle the boring stuff (dirs/more instance variables).
        super(DynamicBot, self).__init__(self.log, dataset, params)
        self.build_computation_graph(dataset)
        self.compile()

    def build_computation_graph(self, dataset):

        # Grab the model classes (Constructors) specified by user in params.
        encoder_class = locate(self.model_params['encoder.class'])
        decoder_class = locate(self.model_params['decoder.class'])
        assert encoder_class is not None, "Couldn't find requested %s." % \
                                          self.model_params['encoder.class']
        assert decoder_class is not None, "Couldn't find requested %s." % \
                                          self.model_params['decoder.class']

        # Create embedder object -- handles all of your embedding needs!
        # By passing scope to embedder calls, we can easily create distinct embeddings,
        # while storing inside the same embedder object.
        self.embedder = Embedder(self.vocab_size, self.embed_size, l1_reg=self.l1_reg)

        # Organize full input pipeline inside single graph node for clean visualization.
        self.pipeline = InputPipeline(dataset.paths,
                                      self.batch_size,
                                      is_chatting=self.is_chatting)
        self.encoder_inputs = self.pipeline.encoder_inputs
        self.decoder_inputs = self.pipeline.decoder_inputs

        with tf.variable_scope("encoder") as scope:
            embedded_enc_inputs = self.embedder(self.encoder_inputs, scope=scope)
            # Create the encoder & decoder objects.
            encoder = encoder_class(self.state_size, self.embed_size,
                                         dropout_prob=self.dropout_prob,
                                         num_layers=self.num_layers)
            # Applying embedded inputs to encoder yields the final (context) state.
            _, encoder_state = encoder(embedded_enc_inputs)

        with tf.variable_scope("decoder") as scope:
            embedded_dec_inputs = self.embedder(self.decoder_inputs, scope=scope)
            self.decoder  = decoder_class(self.state_size,
                                          self.vocab_size,
                                          self.embed_size,
                                          dropout_prob=self.dropout_prob,
                                          num_layers=self.num_layers,
                                          max_seq_len=dataset.max_seq_len,
                                          temperature=self.temperature)
            # For decoder, we want the full sequence of outputs, not simply the last.
            decoder_outputs, decoder_state = self.decoder(embedded_dec_inputs,
                                                          initial_state=encoder_state,
                                                          is_chatting=self.is_chatting,
                                                          loop_embedder=self.embedder,
                                                          scope=scope)

        self.outputs = decoder_outputs
        # Tag inputs and outputs by name should we want to freeze the model.
        self.graph.add_to_collection('freezer', self.encoder_inputs)
        self.graph.add_to_collection('freezer', decoder_outputs)
        # Merge any summaries floating around in the aether into one object.
        self.merged = tf.summary.merge_all()

    def compile(self):
        """ TODO: perhaps merge this into __init__?
        Originally, this function accepted training/evaluation specific parameters.
        However, since moving the configuration parameters to .yaml files and interfacing
        with the dictionary, no args are needed here, and thus would mainly be a hassle
        to have to call before training. Will decide later.
        """

        if not self.is_chatting:
            with tf.variable_scope("evaluation") as scope:
                # Loss - target is to predict, as output, the next decoder input.
                # target_labels has shape [batch_size, dec_inp_seq_len - 1]
                target_labels   = self.decoder_inputs[:, 1:]
                target_weights  = tf.cast(target_labels > 0, target_labels.dtype)
                preds       = self.decoder.apply_projection(self.outputs)
                reg_losses   = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l1          = tf.reduce_sum(tf.abs(reg_losses))

                if self.sampled_loss:
                    self.log.info("Training with dynamic sampled softmax loss.")
                    assert 0 < self.num_samples < self.vocab_size, \
                        "num_samples is %d but should be between 0 and %d" \
                        % (self.num_samples, self.vocab_size)

                    self.loss = bot_ops.dynamic_sampled_softmax_loss(
                        target_labels,
                        self.outputs[:, :-1, :],
                        self.decoder.get_projection_tensors(),
                        self.vocab_size,
                        num_samples=self.num_samples
                    ) + l1
                else:
                    self.loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=target_labels,
                        logits=preds[:, :-1, :],
                        weights=target_weights
                    ) + l1

                self.log.info("Optimizing with %s." % self.optimizer)
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss, global_step=self.global_step,
                    learning_rate=self.learning_rate,
                    optimizer=self.optimizer,
                    clip_gradients=self.max_gradient,
                    summaries=['gradients'])

                # Compute accuracy, ensuring we use fully projected outputs.
                with self.graph.device('/cpu:0'):
                    correct_pred = tf.equal(tf.argmax(preds[:, :-1, :], axis=2),
                                            target_labels)
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('loss_train', self.loss)
                self.merged = tf.summary.merge_all()
                # Note: Important not to merge in the validation loss, don't want to
                # store the training loss on accident.
                self.valid_summ = tf.summary.scalar('loss_valid', self.loss)

        super(DynamicBot, self).compile()

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
            fetches = [self.merged, self.loss, self.train_op]
            summaries, step_loss, _ = self.sess.run(fetches)
            return summaries, step_loss, None
        elif self.is_chatting:
            response = self.sess.run(self.outputs, feed_dict=self.pipeline.feed_dict)
            return None, None, response
        else:
            fetches = [self.valid_summ, self.loss] # , self.outputs]
            summaries, step_loss = self.sess.run(fetches)
            return summaries, step_loss, None

    def train(self, dataset=None):
        """Train bot on inputs until user types CTRL-C or queues run out of data.

        Args:
            dataset: any instance of the Dataset class. Will be removed soon.
        """

        if dataset is None:
            dataset = self.dataset

        def perplexity(loss): return np.exp(float(loss)) if loss < 300 else float("inf")

        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        # Tell embedder to coordinate with TensorBoard's embedding visualization.
        # This allows us to view words in 3D-projected embedding space (with labels!).
        label_paths = [dataset.paths['from_vocab'], dataset.paths['to_vocab']]
        self.embedder.assign_visualizer(self.file_writer, 'encoder', label_paths[0])
        self.embedder.assign_visualizer(self.file_writer, 'decoder', label_paths[1])

        # Note: Calling sleep() allows sustained GPU utilization across training.
        # Without it, looks like GPU has to wait for data to be enqueued more often.
        print('QUEUE RUNNERS RELEASED.', end=" ")
        for _ in range(3): print('.', end=" "); time.sleep(1)
        print('GO!')

        try:
            avg_loss = avg_step_time = 0.0
            while not coord.should_stop():

                i_step = self.sess.run(self.global_step)

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
                    with self.graph.device('/cpu:0'):
                        summaries, eval_loss, _ = self.step(forward_only=True)
                        self.save(summaries=summaries)
                    self.pipeline.toggle_active()
                    print("\tValidation loss = %.3f" % eval_loss, end="; ")
                    print("val perplexity = %.2f" % perplexity(eval_loss))
                    # Reset the running averages and exit checkpoint.
                    avg_loss = avg_step_time = 0.0

                if i_step >= self.max_steps:
                    print("Maximum step", i_step, "reached. Terminating training.")
                    raise SystemExit

        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            coord.request_stop()
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError. You have run out of data. Get some more.")
            coord.request_stop()
        finally:
            coord.join(threads)
            self.close(save_current=False, rebuild_for_chat=True)

    def chat(self):
        """Alias to decode."""
        self.decode()

    def decode(self):
        """Sets up and manages chat session between bot and user (stdin)."""
        # Make sure params are set to chat values, just in case the user
        # forgot to specify/doesn't know about such things.
        self._set_chat_params()
        # Decode from standard input.
        print("Type \"exit\" to exit.\n")
        sentence = io_utils.get_sentence()
        while sentence != 'exit':
            response = self(sentence)
            print("Robot:", response)
            sentence = io_utils.get_sentence()
        print("Farewell, human.")

    def __call__(self, sentence):
        """This is how we talk to the bot interactively. While
        decode(self) above sets up/manages the chat session, users can also use this
        directly to get responses from the bot, given an input sentence. For example,
            sentence = 'Hi, bot!'
            response = bot(sentence)
        is all that's required for back-and-forth with the bot.

        Args:
            sentence: (str) Input sentence from user.

        Returns:
            response string from bot.
        """
        # Convert input sentence to token-ids.
        encoder_inputs = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(sentence), self.dataset.word_to_idx)

        encoder_inputs = np.array([encoder_inputs[::-1]])
        self.pipeline.feed_user_input(encoder_inputs)
        # Get output sentence from the chatbot.
        _, _, response = self.step(forward_only=True)
        # response has shape [1, response_length] and it's last elemeot is EOS_ID. :)
        return self.dataset.as_words(response[0][:-1])

    def close(self, save_current=True, rebuild_for_chat=True):
        """Before closing, which will freeze our graph to a file,
        rebuild it so that it's ready for chatting when unfreezed,
        to make it easier for the user. Training can still be resumed
        with no issue since it doesn't load frozen models, just ckpts.
        """

        if rebuild_for_chat:
            lr_val = self.learning_rate.eval(session=self.sess)
            tf.reset_default_graph()
            # Gross. Am ashamed:
            self.sess = tf.Session()
            with self.graph.name_scope(tf.GraphKeys.SUMMARIES):
                self.global_step    = tf.Variable(initial_value=0, trainable=False)
                self.learning_rate  = tf.constant(lr_val)
            self._set_chat_params()
            self.build_computation_graph(self.dataset)
            self.compile()
        super(DynamicBot, self).close(save_current=save_current)

    def _set_chat_params(self):
        """Set all training-specific param values to chatting-specific values."""
        # TODO: use __setattr__ instead of this.
        self.__dict__['__params']['model_params']['decode'] = True
        self.__dict__['__params']['model_params']['is_chatting'] = True
        self.__dict__['__params']['model_params']['batch_size'] = 1
        self.__dict__['__params']['model_params']['reset_model'] = False
        assert self.is_chatting and self.decode and not self.reset_model

