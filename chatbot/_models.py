"""Abstract classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import yaml
import random
import subprocess

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.client import device_lib
from utils import io_utils
from chatbot.components import *
from chatbot.globals import DEFAULT_FULL_CONFIG, OPTIMIZERS


def gpu_found():
    """Returns True if tensorflow finds at least 1 GPU."""
    devices = device_lib.list_local_devices()
    return len([x.name for x in devices if x.device_type == 'GPU']) > 0


class Model(object):
    """Superclass of all subsequent model classes.
    """

    def __init__(self, logger, dataset, params):
        """
        Args:
            logger: returned by getLogger & called by subclasses. Passed
                    here so we know what object to use for info/warn/error.
            dataset: object that inherits from data.Dataset.
            params: (dict) user-specified params that override those in
                           DEFAULT_FULL_CONFIG above.
        """

        self.log = logger
        self.__dict__['__params'] = Model.fill_params(dataset, params)

        # Make particularly useful ckpt directories for website configurations.
        if 'website_config' in self.ckpt_dir:
            self.ckpt_dir = Model._build_hparam_path(
                ckpt_dir=self.ckpt_dir,
                num_layers=self.num_layers,
                max_seq_len=self.max_seq_len)
            self.log.info("New ckpt dir:", self.ckpt_dir)

        # Configure gpu options if we are using one.
        if gpu_found():
            self.log.info("GPU Found. Setting allow_growth to True.")
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=gpu_config)
        else:
            self.log.warning("GPU not found. Not recommended for training.")
            self.sess = tf.Session()

        with self.graph.name_scope(tf.GraphKeys.SUMMARIES):
            self.global_step = tf.Variable(initial_value=0, trainable=False)
            self.learning_rate = tf.constant(self.learning_rate)

        # Create ckpt_dir if user hasn't already (if exists, has no effect).
        subprocess.call(['mkdir', '-p', self.ckpt_dir])
        self.projector_config = projector.ProjectorConfig()
        # Good practice to set as None in constructor.
        self.loss = None
        self.file_writer = None
        self.merged = None
        self.train_op = None
        self.saver = None

    def compile(self):
        """ Configure training process and initialize model. Inspired by Keras.

        Either restore model parameters or create fresh ones.
            - Checks if we can both (1) find a checkpoint state, and (2) a
            valid V1/V2 checkpoint path.
            - If we can't, then just re-initialize model with fresh params.
        """

        self.log.info("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(self.ckpt_dir)

        if not self.reset_model and checkpoint_state \
                and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
            print("Reading model parameters from",
                  checkpoint_state.model_checkpoint_path)
            self.file_writer = tf.summary.FileWriter(self.ckpt_dir)
            self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters:\n\t", self.ckpt_dir)
            # Recursively delete all files in output but keep directories.
            subprocess.call([
                'find', self.ckpt_dir, '-type', 'f', '-exec', 'rm', '{}', ';'
            ])
            self.file_writer = tf.summary.FileWriter(self.ckpt_dir)
            # Add operation for calling all variable initializers.
            init_op = tf.global_variables_initializer()
            # Construct saver (adds save/restore ops to all).
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
            # Add the fully-constructed graph to the event file.
            self.file_writer.add_graph(self.sess.graph)
            # Initialize all model variables.
            self.sess.run(init_op)
            # Store model config in ckpt dir for easy loading later.
            with open(os.path.join(self.ckpt_dir, 'config.yml'), 'w') as f:
                yaml.dump(getattr(self, "params"), f, default_flow_style=False)

    def save(self, summaries=None):
        """
        Args:
            summaries: merged summary instance returned by session.run.
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

    def close(self, save_current=True):
        """Call then when training session is terminated.
            - Saves the current model/checkpoint state.
            - Freezes the model into a protobuf file in self.ckpt_dir.
            - Closes context managers for file_writing and session.
        """
        # First save the checkpoint as usual.
        if save_current:
            self.save()
        # Freeze me, for I am infinite.
        self.freeze()
        # Be a responsible bot and close my file writer.
        self.file_writer.close()
        # Formally exit the session, farewell to all.
        self.sess.close()

    @property
    def graph(self):
        return self.sess.graph

    @staticmethod
    def fill_params(dataset, params):
        """For now, essentially just returns (already parsed) params, 
        but placed here in case I want to customize later (likely).
        """
        # Replace (string) specification of dataset with the actual instance.
        params['dataset'] = dataset
        params['dataset_params']['data_name'] = dataset.name
        if params['model_params']['ckpt_dir'] == 'out':
            params['model_params']['ckpt_dir'] += '/'+dataset.name
        # Define alias in case older models still use it.
        params['model_params']['is_chatting'] = params['model_params']['decode']
        return params

    def freeze(self):
        """Useful for e.g. deploying model on website.

        Args: directory containing model ckpt files we'd like to freeze.
        """

        if not tf.get_collection('freezer'):
            self.log.warning('No freezer found. Not saving a frozen model.')
            return

        # Note: output_node_names is only used to tell tensorflow what is can
        # throw away in the frozen graph (e.g. training ops).
        output_node_names = ",".join(
            [t.name.rstrip(':0') for t in tf.get_collection('freezer')])
        self.log.info('Output node names: %r', output_node_names)

        # Save a graph with only the bare necessities for chat sessions.
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess, self.graph.as_graph_def(), output_node_names.split(','))

        output_fname = os.path.join(self.ckpt_dir, "frozen_model.pb")
        with tf.gfile.GFile(output_fname, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        subprocess.call(['cp', self.dataset.paths['vocab'], self.ckpt_dir])

    def __getattr__(self, name):
        if name == 'params':
            camel_case = self.data_name.title().replace('_', '')
            replace_dict = {'dataset': "data."+camel_case}
            return {**self.__dict__['__params'], **replace_dict}
        elif name in DEFAULT_FULL_CONFIG: # Requesting a top-level key.
            return self.__dict__['__params'][name]
        else:
            for k in DEFAULT_FULL_CONFIG.keys():
                if not isinstance(self.__dict__['__params'][k], dict):
                    continue
                if name in self.__dict__['__params'][k]:
                    return self.__dict__['__params'][k][name]
        raise AttributeError(name)

    @staticmethod
    def _build_hparam_path(ckpt_dir, **kwargs):
        """Returns relative path build from args for descriptive checkpointing.

        The new path becomes ckpt_dir appended with directories named by kwargs:
            - If a given kwargs[key] is a string, that is set as the 
              appended dir name.
            - Otherwise, it gets formatted, e.g. for key='learning_rate' it 
              may become 'learning_rate_0_001'

        Returns:
            ckpt_dir followed by sequentially appended directories, 
            named by kwargs.
        """
        kwargs = copy.deepcopy(kwargs)
        new_ckpt_dir = ckpt_dir
        for key in sorted(kwargs):
            if not isinstance(kwargs[key], str):
                dir_name = key + "_" + str(kwargs[key]).replace('.', '_')
            else:
                dir_name = kwargs[key]
            new_ckpt_dir = os.path.join(new_ckpt_dir, dir_name)
        return new_ckpt_dir


class BucketModel(Model):
    """Abstract class. Any classes that extend BucketModel just need to customize their
        graph structure in __init__ and implement the step(...) function.
        The real motivation for making this was to be able to use the true Model
        abstract class for all classes in this directory, bucketed or not, r1.0 or r0.12.
    """

    def __init__(self, logger, buckets, dataset, params):
        self.buckets = buckets
        super(BucketModel, self).__init__(
            logger=logger,
            dataset=dataset,
            params=params)

    def compile(self):
        """ Configure training process. Name was inspired by Keras. <3 """

        if self.losses is None:
            raise ValueError("Tried compiling model before defining losses.")

        print("Configuring training operations. This may take some time . . . ")
        # Note: variables are trainable=True by default.
        params = tf.trainable_variables()
        # train_op will store the parameter (S)GD train_op.
        self.apply_gradients = []
        optimizer = OPTIMIZERS[self.optimizer](self.learning_rate)
        for b in range(len(self.buckets)):
            gradients = tf.gradients(self.losses[b], params)
            # Gradient clipping is actually extremely simple, it basically just
            # checks if L2Norm(gradients) > max_gradient, and if it is,
            # it returns (gradients / L2Norm(gradients)) * max_grad.
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.max_gradient)
            self.apply_gradients.append(optimizer.apply_gradients(
                zip(clipped_gradients, params),global_step=self.global_step))

        super(BucketModel, self).compile()

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

        # Also set any decoder-input-weights to 0 that have PAD
        # as target decoder output.
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




