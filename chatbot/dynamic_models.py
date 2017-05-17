"""Sequence-to-sequence models with dynamic unrolling and faster embedding 
techniques.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import logging
import numpy as np
import tensorflow as tf
from chatbot import components
from chatbot.components import bot_ops, Embedder, InputPipeline
from chatbot._models import Model
from utils import io_utils
from pydoc import locate


class DynamicBot(Model):
    """ General sequence-to-sequence model for conversations. 
    
    Will eventually support beam search, and a wider variety of 
    cell options. At present, supports multi-layer encoder/decoders, 
    GRU/LSTM cells, attention, and dynamic unrolling (online decoding included). 
    
    Additionally, will eventually support biologically inspired mechanisms for 
    learning, such as hebbian-based update rules.
    """

    def __init__(self, dataset, params):
        """Build the model computation graph.
        
        Args:
            dataset: any instance inheriting from data.DataSet.
            params: dictionary of hyperparameters.
                    For supported keys, see DEFAULT_FULL_CONFIG.
                    (chatbot.globals.py)
        """

        self.log = logging.getLogger('DynamicBotLogger')
        # Let superclass handle common bookkeeping (saving/loading/dir paths).
        super(DynamicBot, self).__init__(self.log, dataset, params)
        # Build the model's structural components.
        self.build_computation_graph(dataset)
        # Configure training and evaluation.
        # Note: this is distinct from build_computation_graph for historical
        # reasons, and I plan on refactoring. Initially, I more or less followed
        # the feel of Keras for setting up models, but after incorporating the
        # YAML configuration files, this seems rather unnecessary.
        self.compile()

    def build_computation_graph(self, dataset):
        """Create the TensorFlow model graph. Note that this only builds the 
        structural components, i.e. nothing related to training parameters,
        optimization, etc. 
        
        The main components to be built (in order): 
            1. InputPipeline
            2. Embedder
               - single object shared between encoder/decoder.
               - creates distict embeddings for distinct variable scopes.
            2. Encoder
            3. Decoder
        """

        # Grab the model classes (Constructors) specified by user in params.
        encoder_class = locate(getattr(self, 'encoder.class')) \
                        or getattr(components, getattr(self, 'encoder.class'))
        decoder_class = locate(getattr(self, 'decoder.class')) \
                        or getattr(components, getattr(self, 'decoder.class'))

        assert encoder_class is not None, "Couldn't find requested %s." % \
                                          self.model_params['encoder.class']
        assert decoder_class is not None, "Couldn't find requested %s." % \
                                          self.model_params['decoder.class']

        # Organize input pipeline inside single node for clean visualization.
        self.pipeline = InputPipeline(
            file_paths=dataset.paths,
            batch_size=self.batch_size,
            is_chatting=self.is_chatting)

        # Grab the input feeds for encoder/decoder from the pipeline.
        encoder_inputs = self.pipeline.encoder_inputs
        self.decoder_inputs = self.pipeline.decoder_inputs

        # Create embedder object -- handles all of your embedding needs!
        # By passing scope to embedder calls, we can create distinct embeddings,
        # while storing inside the same embedder object.
        self.embedder = Embedder(
            self.vocab_size,
            self.embed_size,
            l1_reg=self.l1_reg)

        # Explicitly show required parameters for any subclass of
        # chatbot.components.base.RNN (e.g. encoders/decoders).
        # I do this for readability; you can easily tell below which additional
        # params are needed, e.g. for a decoder.
        rnn_params = {
            'state_size': self.state_size,
            'embed_size': self.embed_size,
            'num_layers': self.num_layers,
            'dropout_prob': self.dropout_prob,
            'base_cell': self.base_cell}

        with tf.variable_scope('encoder'):
            embedded_enc_inputs = self.embedder(encoder_inputs)
            # For now, encoders require just the RNN params when created.
            encoder = encoder_class(**rnn_params)
            # Apply embedded inputs to encoder for the final (context) state.
            encoder_outputs, encoder_state = encoder(embedded_enc_inputs)

        with tf.variable_scope("decoder"):
            embedded_dec_inputs = self.embedder(self.decoder_inputs)
            # Sneaky. Would be nice to have a "cleaner" way of doing this.
            if getattr(self, 'attention_mechanism', None) is not None:
                rnn_params['attention_mechanism'] = self.attention_mechanism
            self.decoder = decoder_class(
                encoder_outputs=encoder_outputs,
                vocab_size=self.vocab_size,
                max_seq_len=dataset.max_seq_len,
                temperature=self.temperature,
                **rnn_params)

            # For decoder outpus, we want the full sequence (output sentence),
            # not simply the last.
            decoder_outputs, decoder_state = self.decoder(
                embedded_dec_inputs,
                initial_state=encoder_state,
                is_chatting=self.is_chatting,
                loop_embedder=self.embedder)

        self.outputs = tf.identity(decoder_outputs, name='outputs')
        # Tag inputs and outputs by name should we want to freeze the model.
        tf.add_to_collection('freezer', encoder_inputs)
        tf.add_to_collection('freezer', self.outputs)
        # Merge any summaries floating around in the aether into one object.
        self.merged = tf.summary.merge_all()

    def compile(self):
        """ TODO: perhaps merge this into __init__?
        Originally, this function accepted training/evaluation specific 
        parameters. However, since moving the configuration parameters to .yaml 
        files and interfacing with the dictionary, no args are needed here, 
        and thus would mainly be a hassle to have to call before training. 
        
        Will decide how to refactor this later.
        """

        if not self.is_chatting:
            with tf.variable_scope("evaluation") as scope:
                # Loss - target is to predict (as output) next decoder input.
                # target_labels has shape [batch_size, dec_inp_seq_len - 1]
                target_labels = self.decoder_inputs[:, 1:]
                target_weights = tf.cast(target_labels > 0, target_labels.dtype)
                preds = self.decoder.apply_projection(self.outputs)
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                l1 = tf.reduce_sum(tf.abs(reg_losses))

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
                        num_samples=self.num_samples) + l1
                else:
                    self.loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=target_labels,
                        logits=preds[:, :-1, :],
                        weights=target_weights) + l1
                    # New loss function I'm experimenting with below:
                    # I'm suspicious that it may do the same stuff
                    # under-the-hood as sparse_softmax_cross_entropy,
                    # but I'm doing speed tests/comparisons to make sure.
                    #self.loss = bot_ops.cross_entropy_sequence_loss(
                    #    labels=target_labels,
                    #    logits=preds[:, :-1, :],
                    #    weights=target_weights) + l1

                self.log.info("Optimizing with %s.", self.optimizer)
                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss, global_step=self.global_step,
                    learning_rate=self.learning_rate,
                    optimizer=self.optimizer,
                    clip_gradients=self.max_gradient,
                    summaries=['gradients'])

                # Compute accuracy, ensuring we use fully projected outputs.
                correct_pred = tf.equal(tf.argmax(preds[:, :-1, :], axis=2),
                                        target_labels)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('loss_train', self.loss)
                self.merged = tf.summary.merge_all()
                # Note: Important not to merge in the validation loss, since
                # we don't want to couple it with the training loss summary.
                self.valid_summ = tf.summary.scalar('loss_valid', self.loss)

        super(DynamicBot, self).compile()

    def step(self, forward_only=False):
        """Run one step of the model, which can mean 1 of the following:
            1. forward_only == False. 
               - This means we are training.
               - We do a forward and a backward pass.
            2. self.is_chatting. 
               - We are running a user's input sentence to generate a response.
               - We only do a forward pass to get the response (word IDs).
            3. Otherwise: inference (used for validation)
               - Do a forward pass, but also compute loss(es) and summaries.

        Args:
            forward_only: if True, don't perform backward pass 
            (gradient updates).

        Returns:
            3-tuple: (summaries, step_loss, step_outputs).
            
            Qualifications/details for each of the 3 cases:
            1. If forward_only == False: 
               - This is a training step: 'summaries' are training summaries.
               - step_outputs = None
            2. else if self.is_chatting: 
               - summaries = step_loss = None
               - step_outputs == the bot response tokens
            3. else (validation):
               - This is validation: 'summaries' are validation summaries.
               - step_outputs == None (to reduce computational cost).
        """

        if not forward_only:
            fetches = [self.merged, self.loss, self.train_op]
            summaries, step_loss, _ = self.sess.run(fetches)
            return summaries, step_loss, None
        elif self.is_chatting:
            response = self.sess.run(
                fetches=self.outputs,
                feed_dict=self.pipeline.feed_dict)
            return None, None, response
        else:
            fetches = [self.valid_summ, self.loss]  # , self.outputs]
            summaries, step_loss = self.sess.run(fetches)
            return summaries, step_loss, None

    def train(self, dataset=None):
        """Train bot on inputs until user types CTRL-C or queues run out of data.

        Args:
            dataset: (DEPRECATED) any instance of the Dataset class. 
            Will be removed soon.
        """

        def perplexity(loss):
            return np.exp(float(loss)) if loss < 300 else float("inf")

        if dataset is None:
            dataset = self.dataset

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        # Tell embedder to coordinate with TensorBoard's "Embedddings" tab.
        # This allows us to view words in 3D-projected embedding space.
        self.embedder.assign_visualizers(
            self.file_writer,
            ['encoder', 'decoder'],
            dataset.paths['vocab'])

        # Note: Calling sleep allows sustained GPU utilization across training.
        # Without it, GPU has to wait for data to be enqueued more often.
        print('QUEUE RUNNERS RELEASED.', end=" ")
        for _ in range(3):
            print('.', end=" ");
            time.sleep(1);
            sys.stdout.flush()
        print('GO!')

        try:
            avg_loss = avg_step_time = 0.0
            while not coord.should_stop():

                i_step = self.sess.run(self.global_step)

                start_time = time.time()
                summaries, step_loss, _ = self.step()
                # Calculate running averages.
                avg_step_time += (time.time() - start_time) / self.steps_per_ckpt
                avg_loss += step_loss / self.steps_per_ckpt

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
                    print("Maximum step", i_step, "reached.")
                    raise SystemExit

        except (KeyboardInterrupt, SystemExit):
            print("Training halted. Cleaning up . . . ")
            coord.request_stop()
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError. You have run out of data.")
            coord.request_stop()
        finally:
            coord.join(threads)
            self.close(save_current=False, rebuild_for_chat=True)

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
        """This is how we talk to the bot interactively.
        
        While decode(self) above sets up/manages the chat session, 
        users can also use this directly to get responses from the bot, 
        given an input sentence. 
        
        For example, one could do:
            sentence = 'Hi, bot!'
            response = bot(sentence)
        for a single input-to-response with the bot.

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
        # response has shape [1, response_length].
        # Its last element is the EOS_ID, which we don't show user.
        response = self.dataset.as_words(response[0][:-1])
        if 'UNK' in response:
            response = "I don't know."
        return response

    def chat(self):
        """Alias for decode."""
        self.decode()

    def respond(self, sentence):
        """Alias for __call__. (Suggestion)"""
        return self.__call__(sentence)

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
        """Set training-specific param values to chatting-specific values."""
        # TODO: use __setattr__ instead of this.
        self.__dict__['__params']['model_params']['decode'] = True
        self.__dict__['__params']['model_params']['is_chatting'] = True
        self.__dict__['__params']['model_params']['batch_size'] = 1
        self.__dict__['__params']['model_params']['reset_model'] = False
        self.__dict__['__params']['model_params']['dropout_prob'] = 0.0
        assert self.is_chatting and self.decode and not self.reset_model

