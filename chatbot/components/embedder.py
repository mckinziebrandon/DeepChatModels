import tensorflow as tf
import logging
import numpy as np
from chatbot._models import Model
from utils import io_utils
import time


class Embedder:
    """Acts on tensors with integer elements, embedding them in a higher-dimensional
    vector space. A single Embedder instance can embed both encoder and decoder by
    associating them with distinct scopes. """

    def __init__(self, vocab_size, embed_size, l1_reg=0.0):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.l1_reg = l1_reg
        self._scopes = dict()

    def __call__(self, inputs, reuse=None):
        """Embeds integers in inputs and returns the embedded inputs.

        Args:
          inputs: input tensor of shape [batch_size, max_time].

        Returns:
          Output tensor of shape [batch_size, max_time, embed_size]
        """

        # Ensure inputs has expected rank of 2.
        assert len(inputs.shape) == 2, \
            "Expected inputs rank 2 but found rank %r" % len(inputs.shape)

        scope = tf.get_variable_scope()
        # Parse info from scope input needed for reliable reuse across model.
        if scope is not None:
            scope_name = scope if isinstance(scope, str) else scope.name
            if scope_name not in self._scopes:
                self._scopes[scope_name] = scope
        else:
            self._scopes['embedder_call'] = tf.variable_scope('embedder_call')

        embed_tensor = tf.get_variable(
            name="embed_tensor",
            shape=[self.vocab_size, self.embed_size],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=tf.contrib.layers.l1_regularizer(self.l1_reg))
        embedded_inputs = tf.nn.embedding_lookup(embed_tensor, inputs)
        # Place any checks on inputs here before returning.
        if not isinstance(embedded_inputs, tf.Tensor):
            raise TypeError("Embedded inputs should be of type Tensor.")
        if len(embedded_inputs.shape) != 3:
            raise ValueError("Embedded sentence has incorrect shape.")
        tf.summary.histogram(scope.name, embed_tensor)
        return embedded_inputs

    def assign_visualizers(self, writer, scope_names, metadata_path):
        """Setup the tensorboard embedding visualizer.

        Args:
            writer: instance of tf.summary.FileWriter
            scope_names: list of 
        """
        assert writer is not None

        if not isinstance(scope_names, list):
            scope_names = [scope_names]

        for scope_name in scope_names:
            assert scope_name in self._scopes, \
                "I don't have any embedding tensors for %s" % scope_name
            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            emb = config.embeddings.add()
            emb.tensor_name = scope_name.rstrip('/') + '/embed_tensor:0'
            emb.metadata_path = metadata_path
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    def get_scope_basename(self, scope):
        """
        Args:
            scope: tf.variable_scope.
        """
        return scope.name.strip('/').split('/')[-1]


class AutoEncoder(Model):
    """[UNDER CONSTRUCTION]. AutoEncoder for unsupervised pretraining the
    word embeddings for dynamic models.
    """

    def __init__(self, dataset, params):

        self.log = logging.getLogger('AutoEncoderLogger')
        super(AutoEncoder, self).__init__(self.log, dataset, params)
        self.build_computation_graph(dataset)
        self.compile()

    def build_computation_graph(self, dataset):
        from chatbot.components.input_pipeline import InputPipeline
        # Organize input pipeline inside single node for clean visualization.
        self.pipeline = InputPipeline(
            file_paths=dataset.paths,
            batch_size=self.batch_size,
            is_chatting=self.is_chatting)

        self.encoder_inputs = self.pipeline.encoder_inputs

        with tf.variable_scope('autoencoder_encoder'):
            embed_tensor = tf.get_variable(
                name="embed_tensor",
                shape=[self.vocab_size, self.embed_size])
            _h = tf.nn.embedding_lookup(embed_tensor, self.encoder_inputs)
            h = tf.contrib.keras.layers.Dense(self.embed_size, activation='relu')(_h)

        with tf.variable_scope('autoencoder_decoder'):
            w = tf.get_variable(
                name="w",
                shape=[self.embed_size, self.vocab_size],
                dtype=tf.float32)
            b = tf.get_variable(
                name="b",
                shape=[self.vocab_size],
                dtype=tf.float32)

            # Swap 1st and 2nd indices to match expected input of map_fn.
            seq_len = tf.shape(h)[1]
            st_size = tf.shape(h)[2]
            time_major_outputs = tf.reshape(h, [seq_len, -1, st_size])
            # Project batch at single timestep from state space to output space.
            def proj_op(h_t):
                return tf.matmul(h_t, w) + b
            decoder_outputs = tf.map_fn(proj_op, time_major_outputs)
            decoder_outputs = tf.reshape(decoder_outputs,
                                         [-1, seq_len, self.vocab_size])

        self.outputs = tf.identity(decoder_outputs, name='outputs')
        # Tag inputs and outputs by name should we want to freeze the model.
        self.graph.add_to_collection('freezer', self.encoder_inputs)
        self.graph.add_to_collection('freezer', self.outputs)
        # Merge any summaries floating around in the aether into one object.
        self.merged = tf.summary.merge_all()

    def compile(self):

        if not self.is_chatting:
            with tf.variable_scope("evaluation") as scope:
                target_labels = self.encoder_inputs[:, 1:]
                target_weights = tf.cast(target_labels > 0, target_labels.dtype)
                print('\ntl\n', target_labels)
                print('\ntw\n', target_weights)
                preds = self.outputs[:, :-1, :]
                print('\npreds\n', preds)

                self.loss = tf.losses.sparse_softmax_cross_entropy(
                    labels=target_labels,
                    logits=preds,
                    weights=target_weights)
                print(self.loss)

                self.train_op = tf.contrib.layers.optimize_loss(
                    loss=self.loss, global_step=self.global_step,
                    learning_rate=self.learning_rate,
                    optimizer='Adam',
                    summaries=['gradients'])

                # Compute accuracy, ensuring we use fully projected outputs.
                _preds = tf.argmax(self.outputs[:, :-1, :], axis=2)
                correct_pred = tf.equal(
                    _preds,
                    target_labels)
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                tf.summary.scalar('accuracy', accuracy)
                tf.summary.scalar('loss_train', self.loss)
                self.merged = tf.summary.merge_all()
        super(AutoEncoder, self).compile()

    def step(self, forward_only=False):
        if not forward_only:
            return self.sess.run([self.merged, self.loss, self.train_op])
        else:
            return self.sess.run(fetches=tf.argmax(self.outputs[:, :-1, :], axis=2),
                                 feed_dict=self.pipeline.feed_dict)

    def train(self, close_when_done=True):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            avg_loss = avg_step_time = 0.0
            while not coord.should_stop():

                i_step = self.sess.run(self.global_step)
                start_time = time.time()
                summaries, step_loss,  _ = self.step()
                avg_step_time += (time.time() - start_time) / self.steps_per_ckpt
                avg_loss += step_loss / self.steps_per_ckpt

                # Print updates in desired intervals (steps_per_ckpt).
                if i_step % self.steps_per_ckpt == 0:
                    print('loss:', avg_loss)
                    self.save(summaries=summaries)
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
            if close_when_done:
                self.close()

    def __call__(self, sentence):
        encoder_inputs = io_utils.sentence_to_token_ids(
            tf.compat.as_bytes(sentence), self.dataset.word_to_idx)
        encoder_inputs = np.array([encoder_inputs[::-1]])
        self.pipeline.feed_user_input(encoder_inputs)
        # Get output sentence from the chatbot.
        response = self.step(forward_only=True)
        return self.dataset.as_words(response[0])

