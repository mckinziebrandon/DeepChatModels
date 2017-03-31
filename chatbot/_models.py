"""Abstract classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from chatbot.components import *

from utils import io_utils

OPTIMIZERS = {
    'Adagrad':  tf.train.AdagradOptimizer,
    'Adam':     tf.train.AdamOptimizer,
    'SGD':      tf.train.GradientDescentOptimizer,
    'RMSProp':  tf.train.RMSPropOptimizer,
}

# Default values for parameters that could be used by a model, training or otherwise.
DEFAULT_PARAMS = {
    "ckpt_dir": "out",
    "data_dir": "data",
    "dataset": "cornell",
    "decode": False,
    "batch_size": 64,
    "dropout_prob": 0.2,
    "state_size": 512,
    "embed_size": 64,
    "learning_rate": 0.01,
    "l1_reg": 1e-6,
    "lr_decay": 0.98,
    "max_gradient": 5.0,
    "num_layers": 3,
    "num_samples": 512,
    "optimizer": "Adam",
    "reset_model": True,
    "sampled_loss": False,
    "steps_per_ckpt": 200,
    "temperature": 0.0,
}


class Model(object):
    """Superclass of all subsequent model classes.
    """

    def __init__(self, logger, dataset, model_params):

        self.__dict__['__params'] = Model.fill_params(dataset, model_params)
        self.log    = logger
        self.sess   = tf.Session()
        with self.graph.name_scope(tf.GraphKeys.SUMMARIES):
            self.global_step    = tf.Variable(initial_value=0, trainable=False)
            self.learning_rate  = tf.constant(self.learning_rate)
        os.popen('mkdir -p %s' % self.ckpt_dir)  # Just in case :)
        self.projector_config = projector.ProjectorConfig()
        # Good practice to set as None in constructor.
        self.file_writer        = None
        self.train_op    = None
        self.saver              = None

    def compile(self):
        """ Configure training process and initialize model. Inspired by Keras.

        Either restore model parameters or create fresh ones.
            - Checks if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
            - If we can't, then just re-initialize model with fresh params.
        """
        print("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(self.ckpt_dir)
        # Note: If you want to prevent from loading models trained on different dataset,
        # you should store them in their own out/dataname folder, and pass that as the ckpt_dir to config.
        if not self.reset_model and checkpoint_state \
                and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
            print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
            self.file_writer    = tf.summary.FileWriter(self.ckpt_dir)
            self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters:\n", self.ckpt_dir)
            # Recursively delete all files in output but keep directories.
            os.popen("find {0}".format(self.ckpt_dir)+" -type f -exec rm {} \;")
            self.file_writer    = tf.summary.FileWriter(self.ckpt_dir)
            # Add operation for calling all variable initializers.
            init_op = tf.global_variables_initializer()
            # Construct saver (adds save/restore ops to all).
            self.saver = tf.train.Saver(tf.global_variables())
            # Add the fully-constructed graph to the event file.
            self.file_writer.add_graph(self.sess.graph)
            # Initialize all model variables.
            self.sess.run(init_op)

    def save(self, summaries=None):
        """
        Args:
            summaries: merged summary instance returned by session.run.
            save_dir: where to save checkpoints. defaults to self.ckpt_dir.
        """

        if self.saver is None:
            raise ValueError("Tried saving model before defining a saver.")
        ckpt_fname = os.path.join(self.ckpt_dir, "{}.ckpt".format(self.data_name))
        # Saves the state of all global variables in a ckpt file.
        self.saver.save(self.sess, ckpt_fname, global_step=self.global_step)
        if summaries is not None:
            self.file_writer.add_summary(summaries, self.global_step.eval(self.sess))
        else:
            self.log.info("Save called without summaries.")

    def close(self):
        """Call then when training session is terminated."""
        # First save the checkpoint as usual.
        self.save()
        # Freeze me, for I am infinite.
        #Model.freeze_model(self.ckpt_dir)
        self.file_writer.close()
        self.sess.close()

    @property
    def graph(self):
        return self.sess.graph

    @staticmethod
    def fill_params(dataset, model_params):
        """Assigns default values from DEFAULT_PARAMS for keys not in model_params."""

        filled_params = {**DEFAULT_PARAMS, **model_params}
        filled_params['max_seq_len']    = dataset.max_seq_len
        filled_params['vocab_size']     = dataset.vocab_size
        filled_params['data_name']      = dataset.name
        filled_params['dataset']        = dataset # get...this...outta here...
        filled_params['is_chatting']    = filled_params['decode']
        return filled_params

    @staticmethod
    def freeze_model(model_dir):
        """ Useful for e.g. deploying model on website.

        Args: directory containing model ckpt files we'd like to freeze.
        """

        # TODO: Need to ensure batch size set to 1 before freezing.
        model_dir = os.path.abspath(model_dir)
        checkpoint_state  = tf.train.get_checkpoint_state(model_dir)
        assert checkpoint_state is not None, "No ckpt files in %s." % model_dir
        frozen_file = os.path.join(model_dir, "frozen_model.pb")

        print("mod", checkpoint_state.model_checkpoint_path)
        saver = tf.train.import_meta_graph(checkpoint_state.model_checkpoint_path+'.meta',
                                           clear_devices=True)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_state.model_checkpoint_path)
            freezer = tf.get_collection('freezer')

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                freezer)
            with tf.gfile.GFile(frozen_file, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

    def __getattr__(self, name):
        if name not in self.__dict__['__params']:
            raise AttributeError(name)
        else:
            return self.__dict__['__params'][name]


class BucketModel(Model):
    """Abstract class. Any classes that extend BucketModel just need to customize their
        graph structure in __init__ and implement the step(...) function. The real motivation for
        making this was to be able to use the true Model abstract class for all classes in this
        directory, bucketed or not, r1.0 or r0.12.
    """

    def __init__(self, logger, buckets, dataset, model_params):
        super(BucketModel, self).__init__(logger, dataset, model_params)
        self.buckets = buckets

    def compile(self, optimizer=None, max_gradient=5.0, reset=False):
        """ Configure training process. Name was inspired by Keras. <3 """

        if self.losses is None:
            raise ValueError("Tried compiling model before defining losses.")

        print("Configuring training operations. This may take some time . . . ")
        # Note: variables are trainable=True by default.
        params = tf.trainable_variables()
        # train_op will store the parameter (S)GD train_op.
        self.apply_gradients = []
        if optimizer is None:
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        # TODO: Think about how this could optimized, main bottleneck for BucketModels.
        for b in range(len(self.buckets)):
            gradients = tf.gradients(self.losses[b], params)
            # Gradient clipping is actually extremely simple, it basically just
            # checks if L2Norm(gradients) > max_gradient, and if it is, it returns
            # (gradients / L2Norm(gradients)) * max_grad.
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient)
            self.apply_gradients.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                          global_step=self.global_step))

        super(BucketModel, self).compile(reset=reset)

    def check_input_lengths(self, inputs, expected_lengths):
        """
        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        for input, length in zip(inputs, expected_lengths):
            if len(input) != length:
                raise ValueError("Input length doesn't match bucket size:"
                                 " %d != %d." % (len(input), length))

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        Args:
          data: tuple of len(self.buckets). data[bucket_id] == [source_ids, target_ids]
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
            # BasicEncoder inputs are padded and then reversed.
            encoder_pad = [io_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # DynamicDecoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad= [io_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
            decoder_inputs.append([io_utils.GO_ID] + decoder_input + decoder_pad)

        # Define some small helper functions before we re-index & weight.
        def inputs_to_unit(uid, inputs):
            """ Return re-indexed version of inputs array. Description in params below.
            :param uid: index identifier for input timestep/unit/node of interest.
            :param inputs:  single batch of data; inputs[i] is i'th sentence.
            :return:        re-indexed version of inputs as numpy array.
            """
            return np.array([inputs[i][uid] for i in range(self.batch_size)], dtype=np.int32)

        batch_encoder_inputs = [inputs_to_unit(i, encoder_inputs) for i in range(encoder_size)]
        batch_decoder_inputs = [inputs_to_unit(i, decoder_inputs) for i in range(decoder_size)]
        batch_weights        = list(np.ones(shape=(decoder_size, self.batch_size), dtype=np.float32))

        # Set weight for the final decoder unit to 0.0 for all batches.
        for i in range(self.batch_size):
            batch_weights[-1][i] = 0.0

        # Also set any decoder-input-weights to 0 that have PAD as target decoder output.
        for unit_id in range(decoder_size - 1):
            ids_with_pad_target = [b for b in range(self.batch_size)
                                   if decoder_inputs[b][unit_id+1] == io_utils.PAD_ID]
            batch_weights[unit_id][ids_with_pad_target] = 0.0

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def train(self, dataset):
        """ Train chatbot. """
        from chatbot.legacy._train import train
        train(self, dataset)

    def decode(self):
        """ Create chat session between user & chatbot. """
        from chatbot.legacy._decode import decode
        decode(self)

    def step(self, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False):
        """Run a step of the model.

        Args:
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
        """
        raise NotImplemented






































