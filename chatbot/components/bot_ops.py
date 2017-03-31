"""Custom TF 'ops' as meant in the TensorFlow definition of ops."""

import numpy as np
import tensorflow as tf
from utils import io_utils


def dynamic_sampled_softmax_loss(labels, logits, output_projection, vocab_size,
                                 from_scratch=False, num_samples=512, name=None):
    """Sampled softmax loss function able to accept 3D Tensors as input,
       as opposed to the official TensorFlow support for <= 2D. This is
       dynamic because it can be applied across variable-length sequences,
       which are unspecified at initialization with size 'None'.

       Args:
        labels: 2D integer tensor of shape [batch_size, None] containing
            the word ID labels for each individual rnn state from logits.
        logits: 3D float tensor of shape [batch_size, None, state_size] as
            ouput by a DynamicDecoder instance.
        from_scratch: (bool) Whether to use the version I wrote from scratch, or to use
                      the version I wrote that applies map_fn(sampled_softmax) across timeslices, which
                      is probably less efficient. (Currently testing)
        num
        Returns:
            loss as a scalar Tensor, computed as the mean over all batches and sequences.
    """

    if from_scratch:
        return _dynamic_sampled_from_scratch(labels, logits, output_projection, vocab_size,
                                             num_samples=num_samples, name=name)
    else:
        return _dynamic_sampled_map(labels, logits, output_projection, vocab_size,
                                    num_samples=num_samples, name=name)


def _dynamic_sampled_map(labels, logits, output_projection, vocab_size,
                                 num_samples=512, name=None):
    """Sampled softmax loss function able to accept 3D Tensors as input,
       as opposed to the official TensorFlow support for <= 2D. This is
       dynamic because it can be applied across variable-length sequences,
       which are unspecified at initialization with size 'None'.

       Args:
           labels: 2D integer tensor of shape [batch_size, None] containing
                the word ID labels for each individual rnn state from logits.
            logits: 3D float tensor of shape [batch_size, None, state_size] as
                ouput by a DynamicDecoder instance.

        Returns:
            loss as a scalar Tensor, computed as the mean over all batches and sequences.
    """
    with tf.name_scope(name, "dynamic_sampled_softmax_loss", [labels, logits, output_projection]):
        seq_len  = tf.shape(logits)[1]
        st_size  = tf.shape(logits)[2]
        time_major_outputs  = tf.reshape(logits, [seq_len, -1, st_size])
        time_major_labels   = tf.reshape(labels, [seq_len, -1])
        # Reshape is apparently faster (dynamic) than transpose.
        w_t = tf.reshape(output_projection[0], [vocab_size, -1])
        b = output_projection[1]
        def sampled_loss(elem):
            logits, lab = elem
            lab = tf.reshape(lab, [-1, 1])
            # TODO: Figure out how this accurately gets loss without requiring weights,
            # like sparse_softmax_cross_entropy requires.
            return tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=w_t,
                    biases=b,
                    labels=lab,
                    inputs=logits,
                    num_sampled=num_samples,
                    num_classes=vocab_size,
                    partition_strategy='div'))
        batch_losses = tf.map_fn(sampled_loss,
                                 (time_major_outputs, time_major_labels),
                                 dtype=tf.float32)
        loss = tf.reduce_mean(batch_losses)
    return loss


def _dynamic_sampled_from_scratch(labels, logits, output_projection, vocab_size,
                                  num_samples, name=None):
    """Note: I closely follow the notation from Tensorflow's Candidate Sampling reference.
       - Link: https://www.tensorflow.org/extras/candidate_sampling.pdf

    Args:
        output_projection: (tuple) returned by any DynamicDecoder.get_projections_tensors()
            - output_projection[0] == w tensor. [state_size, vocab_size]
            - output_projection[0] == b tensor. [vocab_size]
        labels: 2D Integer tensor. [batch_size, None]
        logits: 3D float Tensor [batch_size, None, state_size].
            - In this project, usually is the decoder batch output sequence (NOT projected).
        num_samples: number of classes out of vocab_size possible to use.
        vocab_size: total number of classes.
    """
    with tf.name_scope(name, "dynamic_sampled_from_scratch", [labels, logits, output_projection]):
        batch_size, seq_len, state_size  = tf.unstack(tf.shape(logits))
        time_major_outputs  = tf.reshape(logits, [seq_len, batch_size, state_size])
        time_major_labels   = tf.reshape(labels, [seq_len, batch_size])

        weights = tf.transpose(output_projection[0])
        biases = output_projection[1]
        def sampled_loss_single_timestep(args):
            """
            Args: 2-tuple (because map_fn below)
                targets: 1D tensor (sighs loudly) of shape [batch_size]
                logits: 2D tensor (sighs intensify) of shape [batch_size, state_size].
            """
            logits, targets = args
            with tf.name_scope("compute_sampled_logits", [weights, biases, logits, targets]):
                targets = tf.cast(targets, tf.int64)
                sampled_values = tf.nn.log_uniform_candidate_sampler(
                    true_classes=tf.expand_dims(targets, -1), num_true=1, num_sampled=num_samples,
                    unique=True, range_max=vocab_size)
                S, Q_true, Q_samp = (tf.stop_gradient(s) for s in sampled_values)

                # Get concatenated 1D tensor of shape [batch_size * None + num_samples],
                all_ids = tf.concat([targets, S], 0)
                _W       = tf.nn.embedding_lookup(weights, all_ids, partition_strategy='div')
                _b       = tf.nn.embedding_lookup(biases, all_ids)

                W = {'targets': tf.slice(_W, begin=[0, 0], size=[batch_size, state_size]),
                     'samples': tf.slice(_W, begin=[batch_size, 0], size=[num_samples, state_size])}
                b = {'targets': tf.slice(_b, begin=[0], size=[batch_size]),
                     'samples': tf.slice(_b, begin=[batch_size], size=[num_samples])}

                true_logits  = tf.reduce_sum(tf.multiply(logits, W['targets']), 1)
                true_logits += b['targets'] - tf.log(Q_true)

                sampled_logits  = tf.matmul(logits, W['samples'], transpose_b=True)
                sampled_logits += b['samples'] - tf.log(Q_samp)

                F = tf.concat([true_logits, sampled_logits], 1)
                def fn(s_i): return tf.where(targets == s_i, tf.ones_like(targets), tf.zeros_like(targets))
                sample_labels = tf.transpose(tf.map_fn(fn, S))
                out_targets = tf.concat([tf.ones_like(true_logits, dtype=tf.int64), sample_labels], 1)
            return tf.losses.softmax_cross_entropy(out_targets, logits=F)

        return tf.reduce_mean(tf.map_fn(sampled_loss_single_timestep,
                                        (time_major_outputs, time_major_labels),
                                        dtype=tf.float32))


def sentence_to_inputs(sentence, word_to_idx):
    """Purpose: make an op for translating stdin sentence input from user
    into input suitable for **frozen** model in particular.
    """
    with tf.name_scope("sentence_to_inputs",
                       values=[sentence, word_to_idx]) as scope:
        encoder_inputs = io_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence),
                                                        word_to_idx)
        encoder_inputs = np.array([encoder_inputs[::-1]])
        return tf.convert_to_tensor(encoder_inputs, name=scope)

