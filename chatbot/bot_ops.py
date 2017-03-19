"""Custom TF 'ops' as meant in the TensorFlow definition of ops."""

import tensorflow as tf


def dynamic_sampled_softmax_loss(labels, logits, output_projection, vocab_size,
                                 num_samples=512, name=None):
    """Sampled softmax loss function able to accept 3D Tensors as input,
       as opposed to the official TensorFlow support for <= 2D. This is
       dynamic because it can be applied across variable-length sequences,
       which are unspecified at initialization with size 'None'.

       Args:
           labels: 2D integer tensor of shape [batch_size, None] containing
                the word ID labels for each individual rnn state from logits.
            logits: 3D float tensor of shape [batch_size, None, state_size] as
                ouput by a Decoder instance.

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
            # TODO: Figure out how this accurately gets loss without requiring weights,
            # like sparse_softmax_cross_entropy requires.
            return tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=w_t,
                    biases=b,
                    labels=tf.expand_dims(lab, -1),
                    inputs=logits,
                    num_sampled=num_samples,
                    num_classes=vocab_size,
                    partition_strategy='div'))
        batch_losses = tf.map_fn(sampled_loss, (time_major_outputs, time_major_labels), dtype=tf.float32)
        loss = tf.reduce_mean(batch_losses)
    return loss