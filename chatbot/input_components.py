import tensorflow as tf
from utils.io_utils import GO_ID
from tensorflow.contrib.training import bucket_by_sequence_length

__all__ = ['InputPipeline', 'Embedder']

LENGTHS = {'encoder_sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
           'decoder_sequence_length': tf.FixedLenFeature([], dtype=tf.int64)}
SEQUENCES = {'encoder_sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64),
             'decoder_sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64)}


class InputPipeline:
    """ [NEW] TensorFlow-only input pipeline with parallel enqueuing,
        dynamic bucketed-batching, and more.

        Overview of pipeline construction:
            1. Create ops for reading protobuf tfrecords line-by-line.
            2. Enqueue raw outputs, attach to threads, and parse sequences.
            3. Organize sequences into buckets of similar lengths, pad, and batch.
    """

    def __init__(self, file_paths, batch_size, capacity=None, is_chatting=False, scope=None):
        """
        Args:
            file_paths: (dict) returned by instance of Dataset via Dataset.paths.
            batch_size: number of examples returned by dequeue op.
            capacity: maximum number of examples allowed in the input queue at a time.
            is_chatting: (bool) determines whether we're feeding user input or file inputs.
        """
        with tf.variable_scope(scope, 'input_pipeline'):
            if capacity is None:
                self.capacity = 20 * batch_size
            self.batch_size = batch_size
            self.paths = file_paths
            self.num_threads = 4
            self.control = {'train': 0, 'valid': 1}
            self.active_data = tf.convert_to_tensor(self.control['train'])
            self.is_chatting = is_chatting
            self._user_input = tf.placeholder(tf.int32, [1, None], name='user_input_ph')
            self._feed_dict = None

            if not is_chatting:
                # Create tensors that will store input batches at runtime.
                self._train_lengths, self.train_batches = self.build_pipeline('train')
                self._valid_lengths, self.valid_batches = self.build_pipeline('valid')

    def build_pipeline(self, name):
        """Creates a new input subgraph composed of the following components:
            - Reader queue that feeds protobuf data files.
            - RandomShuffleQueue assigned parallel-thread queuerunners.
            - Dynamic padded-bucketed-batching queue for organizing batches in a time and
              space-efficient manner.

        Args:
            name: filename prefix for desired set of data. See Dataset class for naming conventions.

        Returns:
            2-tuple (lengths, sequences):
                lengths: (dict) parsed context feature from protobuf file.
                Supports keys in LENGTHS.
                sequences: (dict) parsed feature_list from protobuf file.
                Supports keys in SEQUENCES.
        """
        with tf.variable_scope(name + '_pipeline'):
            proto_text = self._read_line(self.paths[name + '_tfrecords'])
            context_pair, sequence_pair = self._assign_queue(proto_text)
            input_length = tf.add(context_pair['encoder_sequence_length'],
                                  context_pair['decoder_sequence_length'],
                                  name=name + 'length_add')
            return self._padded_bucket_batches(input_length, sequence_pair)

    @property
    def encoder_inputs(self):
        """Determines, via tensorflow control structures, which part of the pipeline to run
           and retrieve inputs to a Model encoder component. """
        if not self.is_chatting:
            return self._cond_input('encoder')
        else:
            return self._user_input

    @property
    def decoder_inputs(self):
        """Determines, via tensorflow control structures, which part of the pipeline to run
           and retrieve inputs to a Model decoder component. """
        if not self.is_chatting:
            return self._cond_input('decoder')
        else:
            # In a chat session, we just give the bot the go-ahead to respond!
            return tf.convert_to_tensor([[GO_ID]])

    @property
    def feed_dict(self):
        return self._feed_dict

    def feed_user_input(self, user_input):
        """Called by Model instances upon receiving input from stdin."""
        self._feed_dict = {self._user_input.name: user_input}

    def toggle_active(self):
        """Simple callable that toggles the input data pointer between training and validation."""
        def to_valid(): return tf.constant(self.control['valid'])
        def to_train(): return tf.constant(self.control['train'])
        self.active_data = tf.cond(tf.equal(self.active_data, self.control['train']),
                                    to_valid, to_train)

    def _cond_input(self, prefix):
        with tf.variable_scope(prefix + '_cond_input'):
            def train(): return self.train_batches[prefix + '_sequence']
            def valid(): return self.valid_batches[prefix + '_sequence']
            return tf.cond(tf.equal(self.active_data, self.control['train']),
                           train, valid)

    def _read_line(self, file):
        """Create ops for extracting lines from files.

        Returns:
            Tensor that will contain the lines at runtime.
        """
        with tf.variable_scope('reader'):
            tfrecords_fname = file
            filename_queue = tf.train.string_input_producer([tfrecords_fname])
            reader = tf.TFRecordReader(name='tfrecord_reader')
            _, next_raw = reader.read(filename_queue, name='read_records')
        return next_raw

    def _assign_queue(self, data):
        """
        Args:
            data: object to be enqueued and managed by parallel threads.
        """

        with tf.variable_scope('shuffle_queue'):
            queue = tf.RandomShuffleQueue(capacity=self.capacity,
                                          min_after_dequeue=3*self.batch_size,
                                          dtypes=tf.string, shapes=[()])
            enqueue_op = queue.enqueue(data)
            example_dq = queue.dequeue()
            qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
            tf.train.add_queue_runner(qr)
            _sequence_lengths, _sequences = tf.parse_single_sequence_example(
                serialized=example_dq, context_features=LENGTHS, sequence_features=SEQUENCES)
        return _sequence_lengths, _sequences

    def _padded_bucket_batches(self, input_length, data):
        with tf.variable_scope('bucket_batch'):
            lengths, sequences = bucket_by_sequence_length(
                input_length=tf.to_int32(input_length),
                tensors=data,
                batch_size=self.batch_size,
                bucket_boundaries=[16, 32, 64, 128],
                capacity=self.capacity,
                dynamic_pad=True,
            )
        return lengths, sequences


class Embedder:
    """Acts on tensors with integer elements, embedding them in a higher-dimensional
    vector space. A single Embedder instance can embed both encoder and decoder by associating them with
    distinct scopes. """

    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

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
                                     regularizer=tf.contrib.layers.l1_regularizer(0.1))
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
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        emb = config.embeddings.add()
        with tf.variable_scope(scope, reuse=True):
            emb.tensor_name = tf.get_variable("embed_tensor").name
        emb.metadata_path = metadata_path
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

