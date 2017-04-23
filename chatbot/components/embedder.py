import tensorflow as tf


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
