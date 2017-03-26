import tensorflow as tf


class Embedder:
    """Acts on tensors with integer elements, embedding them in a higher-dimensional
    vector space. A single Embedder instance can embed both encoder and decoder by associating them with
    distinct scopes. """

    def __init__(self, vocab_size, embed_size, l1_reg=0.0):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.l1_reg = l1_reg

    def __call__(self, inputs, scope=None, reuse=None):
        """Embeds integers in inputs and returns the embedded inputs.

        Args:
          inputs: input tensor of shape [batch_size, max_time].

        Returns:
          Output tensor of shape [batch_size, max_time, embed_size]
        """
        assert len(inputs.shape) == 2, "Expected inputs rank 2 but found rank %r" % len(inputs.shape)
        with tf.variable_scope(scope, "embedding_inputs", values=[inputs], reuse=reuse):
            params = tf.get_variable("embed_tensor", [self.vocab_size, self.embed_size],
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     regularizer=tf.contrib.layers.l1_regularizer(self.l1_reg))
            embedded_inputs = tf.nn.embedding_lookup(params, inputs)
            if not isinstance(embedded_inputs, tf.Tensor):
                raise TypeError("Embedded inputs should be of type Tensor.")
            if len(embedded_inputs.shape) != 3:
                raise ValueError("Embedded sentence has incorrect shape.")
            tf.summary.histogram('embedding_encoder', params)
        return embedded_inputs

    def assign_visualizer(self, writer, scope, metadata_path):
        """Setup the tensorboard embedding visualizer.

        Args:
            writer: instance of tf.summary.FileWriter
        """
        assert(writer is not None)
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        emb = config.embeddings.add()
        with tf.variable_scope(scope, reuse=True):
            emb.tensor_name = tf.get_variable("embed_tensor").name
        emb.metadata_path = metadata_path
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

