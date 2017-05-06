import logging
import tensorflow as tf
from utils import io_utils
from tensorflow.contrib.training import bucket_by_sequence_length

LENGTHS = {'encoder_sequence_length': tf.FixedLenFeature([], dtype=tf.int64),
           'decoder_sequence_length': tf.FixedLenFeature([], dtype=tf.int64)}
SEQUENCES = {'encoder_sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64),
             'decoder_sequence': tf.FixedLenSequenceFeature([], dtype=tf.int64)}


class InputPipeline:
    """TensorFlow-only input pipeline with parallel enqueuing,
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
        with tf.name_scope(scope, 'input_pipeline') as scope:
            if capacity is None:
                self.capacity = max(50 * batch_size, int(1e4))
                logging.info("Input capacity set to %d examples." % self.capacity)
            self.batch_size = batch_size
            self.paths = file_paths
            self.control = {'train': 0, 'valid': 1}
            self.active_data = tf.convert_to_tensor(self.control['train'])
            self.is_chatting = is_chatting
            self._user_input = tf.placeholder(tf.int32, [1, None], name='user_input')
            self._feed_dict = None
            self._scope = scope

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
            name: filename prefix for data. See Dataset class for naming conventions.

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
            return tf.convert_to_tensor([[io_utils.GO_ID]])

    @property
    def user_input(self):
        return self._user_input

    @property
    def feed_dict(self):
        return self._feed_dict

    def feed_user_input(self, user_input):
        """Called by Model instances upon receiving input from stdin."""
        self._feed_dict = {self._user_input.name: user_input}

    def toggle_active(self):
        """Simple callable that toggles active_data between training and validation."""
        def to_valid(): return tf.constant(self.control['valid'])
        def to_train(): return tf.constant(self.control['train'])
        self.active_data = tf.cond(tf.equal(self.active_data, self.control['train']),
                                    to_valid, to_train)

    def _cond_input(self, prefix):
        with tf.name_scope(self._scope):
            def train(): return self.train_batches[prefix + '_sequence']
            def valid(): return self.valid_batches[prefix + '_sequence']
            return tf.cond(tf.equal(self.active_data, self.control['train']),
                           train, valid, name=prefix + '_cond_input')

    def _read_line(self, file):
        """Create ops for extracting lines from files.

        Returns:
            Tensor that will contain the lines at runtime.
        """
        with tf.variable_scope('reader'):
            filename_queue = tf.train.string_input_producer([file])
            reader = tf.TFRecordReader(name='tfrecord_reader')
            _, next_raw = reader.read(filename_queue, name='read_records')
        return next_raw

    def _assign_queue(self, proto_text):
        """
        Args:
            proto_text: object to be enqueued and managed by parallel threads.
        """

        with tf.variable_scope('shuffle_queue'):
            queue = tf.RandomShuffleQueue(
                capacity=self.capacity,
                min_after_dequeue=10*self.batch_size,
                dtypes=tf.string, shapes=[()])

            enqueue_op = queue.enqueue(proto_text)
            example_dq = queue.dequeue()

            qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
            tf.train.add_queue_runner(qr)

            _sequence_lengths, _sequences = tf.parse_single_sequence_example(
                serialized=example_dq,
                context_features=LENGTHS,
                sequence_features=SEQUENCES)
        return _sequence_lengths, _sequences

    def _padded_bucket_batches(self, input_length, data):
        with tf.variable_scope('bucket_batch'):
            lengths, sequences = bucket_by_sequence_length(
                input_length=tf.to_int32(input_length),
                tensors=data,
                batch_size=self.batch_size,
                bucket_boundaries=[8, 16, 32],
                capacity=self.capacity,
                dynamic_pad=True)
        return lengths, sequences

