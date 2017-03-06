"""Abstract parent class for all in models.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard python imports.
import os
import random
import logging

# ML/DL-specific imports.
import numpy as np
import tensorflow as tf

# User-defined imports.
from utils import data_utils
from chatbot._train import train
from chatbot._decode import decode


class Model(object):
    """Superclass of all subsequent model classes.
    """

    def __init__(self,
                 buckets,
                 ckpt_dir="out",
                 vocab_size=40000,
                 batch_size=64,
                 learning_rate=0.5,
                 lr_decay=0.98,
                 is_decoding=False):

        self.sess           = tf.Session()
        self.is_decoding    = is_decoding
        self.batch_size     = batch_size
        self.buckets        = buckets
        self.vocab_size = vocab_size
        self.learning_rate  = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.lr_decay_op    = self.learning_rate.assign(learning_rate * lr_decay)
        self.global_step    = tf.Variable(initial_value=0, trainable=False)
        # Directory IO management.
        self.ckpt_dir = ckpt_dir
        self.log_dir = os.path.join(ckpt_dir, "logs")
        self.file_writer    = tf.summary.FileWriter(self.log_dir)
        # Responsibility of user to determine training operations.
        self.apply_gradients = None
        self.losses = None

    def compile(self, optimizer, max_gradient=5.0):
        """ Configure training process. Name was inspired by Keras. <3 """
        raise NotImplemented

    def initialize(self, reset=False):
        """Either restore model parameters or create fresh ones."""
        raise NotImplemented

    def save(self):
        """TODO"""
        raise NotImplemented

    def train(self, dataset, train_config):
        """ Train chatbot. """
        raise NotImplemented

    def decode(self, test_config):
        """ Create chat session between user & chatbot. """
        raise NotImplemented


class BucketModel(Model):
    """Abstract class. Extended by models that emply bucketing techniques.
    The real motivation for making this was to be able to use the true Model abstract
    class for all classes in this directory, bucketed or not, r1.0 or r0.12.
    """

    def initialize(self, reset=False):
        """Either restore model parameters or create fresh ones.
            - Checks if we can both (1) find a checkpoint state, and (2) a valid V1/V2 checkpoint path.
            - If we can't, then just re-initialize model with fresh params.
        """
        print("Checking for checkpoints . . .")
        checkpoint_state  = tf.train.get_checkpoint_state(self.ckpt_dir)
        # Note: If you want to prevent from loading models trained on different dataset,
        # you should store them in their own out/dataname folder, and pass that as the ckpt_dir to config.
        if not reset  and checkpoint_state \
                and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
            print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
            self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            # Clear output dir contents.
            os.popen('rm -rf out/* && mkdir -p out/logs')
            # Add operation for calling all variable initializers.
            init_op = tf.global_variables_initializer()
            # Construct saver (adds save/restore ops to all).
            self.saver = tf.train.Saver(tf.global_variables())
            # Add the fully-constructed graph to the event file.
            self.file_writer.add_graph(self.sess.graph)
            # Initialize all model variables.
            self.sess.run(init_op)

    def compile(self, optimizer, max_gradient=5.0):
        """ Configure training process. Name was inspired by Keras. <3 """

        if self.losses is None:
            raise ValueError("Tried compiling model before defining losses.")
        print("Configuring training operations. This may take some time . . . ")
        # Note: variables are trainable=True by default.
        params = tf.trainable_variables()
        #if not self.is_decoding: # teacher mode means we always need backward pass option.
        # apply_gradients will store the parameter (S)GD apply_gradients.
        self.apply_gradients = []
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # TODO: Think about how this could optimized. There has to be a way.
        for b in range(len(self.buckets)):
            # Note: tf.gradients returns in form: gradients[i] == sum([dy/dx_i for y in self.losses[b]]).
            gradients = tf.gradients(self.losses[b], params)
            # Gradient clipping is actually extremely simple, it basically just
            # checks if L2Norm(gradients) > max_gradient, and if it is, it returns
            # (gradients / L2Norm(gradients)) * max_grad.
            # norm: literally just L2-norm of gradients.
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient)
            self.apply_gradients.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                          global_step=self.global_step))

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
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad= [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)

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
                                   if decoder_inputs[b][unit_id+1] == data_utils.PAD_ID]
            batch_weights[unit_id][ids_with_pad_target] = 0.0

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights


    def train(self, dataset, train_config):
        """ Train chatbot. """
        self.initialize()
        train(self, dataset, train_config)

    def decode(self, test_config):
        """ Create chat session between user & chatbot. """
        self.initialize()
        decode(self, test_config)




































