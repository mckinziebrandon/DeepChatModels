""" ABC for datasets. """
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import io_utils
import os
import random
from utils.io_utils import EOS_ID, PAD_ID, GO_ID, UNK_ID
import logging

from chatbot.globals import DEFAULT_FULL_CONFIG
DEFAULT_PARAMS = DEFAULT_FULL_CONFIG['dataset_params']


class DatasetABC(metaclass=ABCMeta):

    @abstractmethod
    def train_generator(self, batch_size):
        """Returns a generator function for batches of batch_size train data."""
        pass

    @abstractmethod
    def valid_generator(self, batch_size):
        """Returns a generator function for batches of batch_size validation data."""
        pass

    @abstractproperty
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        pass

    @abstractproperty
    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        pass

    @abstractproperty
    def name(self):
        """Returns name of the dataset as a string."""
        pass

    @abstractproperty
    def max_seq_len(self):
        """Return the number of tokens in the longest example"""
        pass


class Dataset(DatasetABC):

    def __init__(self, dataset_params):
        """Implements the general of subset of operations that all classes can use.

        Args:
            dataset_params: dictionary of configuration parameters. See DEFAULT_FULL_CONFIG
                            at top of file for supported keys.
        """

        self.__dict__['__params'] = Dataset.fill_params(dataset_params)
        print("max_seq_len recorded as ", self.max_seq_len)
        # We query io_utils to ensure all data files are organized properly,
        # and io_utils returns the paths to files of interest.
        paths_triplet = io_utils.prepare_data(self.data_dir,
                                              self.data_dir + "/train_from.txt",
                                              self.data_dir + "/train_to.txt",
                                              self.data_dir + "/valid_from.txt",
                                              self.data_dir + "/valid_to.txt",
                                              self.vocab_size, self.vocab_size)

        train_path, valid_path, vocab_path = paths_triplet
        self.paths = dict()
        self.paths['from_train']    = train_path[0]
        self.paths['to_train']      = train_path[1]
        self.paths['from_valid']    = valid_path[0]
        self.paths['to_valid']      = valid_path[1]
        self.paths['from_vocab']    = vocab_path[0]
        self.paths['to_vocab']      = vocab_path[1]
        self.paths['train_tfrecords'] = None
        self.paths['valid_tfrecords'] = None

        self._word_to_idx, _ = io_utils.get_vocab_dicts(self.paths['from_vocab'])
        _, self._idx_to_word = io_utils.get_vocab_dicts(self.paths['to_vocab'])

        # Create tfrecords file if not located in data_dir.
        self.convert_to_tf_records('train')
        self.convert_to_tf_records('valid')

    def __getattr__(self, name):
        if name not in self.__dict__['__params']:
            raise AttributeError(name)
        else:
            return self.__dict__['__params'][name]

    def convert_to_tf_records(self, prefix='train'):
        """If can't find tfrecords 'prefix' files, creates them.

        Args:
            prefix: 'train' or 'valid'. Determines which tfrecords to build.
        """

        from_path = self.paths['from_'+prefix]
        to_path = self.paths['to_'+prefix]
        output_path = os.path.join(
            self.data_dir, prefix + 'voc%d_seq%d' % (self.vocab_size, self.max_seq_len) + '.tfrecords')
        if os.path.isfile(output_path):
            self.log.info('Using tfrecords file %s' % output_path)
            self.paths[prefix + '_tfrecords'] = output_path
            return

        def get_sequence_example(encoder_line, decoder_line):
            space_needed = max(len(encoder_line.split()), len(decoder_line.split()))
            if space_needed > self.max_seq_len:
                return None

            example  = tf.train.SequenceExample()
            encoder_list = [int(x) for x in encoder_line.split()]
            decoder_list = [io_utils.GO_ID] + [int(x) for x in decoder_line.split()] + [EOS_ID]
            example.context.feature['encoder_sequence_length'].int64_list.value.append(len(encoder_list))
            example.context.feature['decoder_sequence_length'].int64_list.value.append(len(decoder_list))

            encoder_sequence = example.feature_lists.feature_list['encoder_sequence']
            decoder_sequence = example.feature_lists.feature_list['decoder_sequence']
            for e in encoder_list:
                encoder_sequence.feature.add().int64_list.value.append(e)
            for d in decoder_list:
                decoder_sequence.feature.add().int64_list.value.append(d)

            return example

        with tf.gfile.GFile(from_path, mode="r") as encoder_file:
            with tf.gfile.GFile(to_path, mode="r") as decoder_file:
                with tf.python_io.TFRecordWriter(output_path) as writer:
                    encoder_line, decoder_line = encoder_file.readline(), decoder_file.readline()
                    while encoder_line and decoder_line:
                        sequence_example = get_sequence_example(encoder_line, decoder_line)
                        if sequence_example is not None:
                            writer.write(sequence_example.SerializeToString())
                        encoder_line, decoder_line = encoder_file.readline(), decoder_file.readline()

        self.log.info("Converted text files %s and %s into tfrecords file %s" \
                      % (from_path, to_path, output_path))
        self.paths[prefix + '_tfrecords'] = output_path

    def sentence_generator(self):
        """Yields (as words) single sentences from training data, for testing purposes."""
        with tf.gfile.GFile(self.data_dir + '/train_from.txt', mode="r") as text_file:
            sentence = text_file.readline().strip()
            while sentence:
                yield sentence
                sentence = text_file.readline().strip()

    def train_generator(self, batch_size):
        """[Note: not needed by DynamicBot since InputPipeline]"""
        return self._generator(self.paths['from_train'], self.paths['to_train'], batch_size)

    def valid_generator(self, batch_size):
        """[Note: not needed by DynamicBot since InputPipeline]"""
        return self._generator(self.paths['from_valid'], self.paths['to_valid'], batch_size)

    def _generator(self, from_path, to_path, batch_size):
        """Returns a generator function that reads data from file, an d
            yields shuffled batches.

        Args:
            from_path: full path to file for encoder inputs.
            to_path: full path to file for decoder inputs.
            batch_size: number of samples to yield at once.
        """

        def longest_sentence(enc_list, dec_list):
            max_enc_len = max([len(s) for s in enc_list])
            max_dec_len = max([len(s) for s in dec_list])
            return max(max_enc_len, max_dec_len)

        def padded_batch(encoder_tokens, decoder_tokens):
            max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
            encoder_batch = np.array([s + [PAD_ID] * (max_sent_len - len(s)) for s in encoder_tokens])[:, ::-1]
            decoder_batch = np.array([s + [PAD_ID] * (max_sent_len - len(s)) for s in decoder_tokens])
            return encoder_batch, decoder_batch

        encoder_tokens = []
        decoder_tokens = []
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:

                source, target = source_file.readline(), target_file.readline()
                while source and target:

                    # Skip any sentence pairs that are too long for user specifications.
                    space_needed = max(len(source.split()), len(target.split()))
                    if space_needed > self.max_seq_len:
                        source, target = source_file.readline(), target_file.readline()
                        continue

                    # Reformat token strings to token lists.
                    # Note: GO_ID is prepended by the chat bot, since it determines
                    # whether or not it's responsible for responding.
                    encoder_tokens.append([int(x) for x in source.split()])
                    decoder_tokens.append([int(x) for x in target.split()] + [EOS_ID])

                    # Have we collected batch_size number of sentences? If so, pad & yield.
                    assert len(encoder_tokens) == len(decoder_tokens)
                    if len(encoder_tokens) == batch_size:
                        yield padded_batch(encoder_tokens, decoder_tokens)
                        encoder_tokens = []
                        decoder_tokens = []
                    source, target = source_file.readline(), target_file.readline()

                # Don't forget to yield the 'leftovers'!
                assert len(encoder_tokens) == len(decoder_tokens)
                assert len(encoder_tokens) <= batch_size
                if len(encoder_tokens) > 0:
                    yield padded_batch(encoder_tokens, decoder_tokens)
    @property
    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        return self._word_to_idx

    @property
    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        return self._idx_to_word

    def as_words(self, sentence):
        """Initially, this function was just the one-liner below:

            return " ".join([tf.compat.as_str(self._idx_to_word[i]) for i in sentence])

            Since then, it has become apparent that some character aren't converted properly,
            and tf has issues decoding. In (rare) cases that this occurs, I've setup the
            try-catch block to help inspect the root causes. It will remain here until the
            problem has been adequately diagnosed.
        """
        words = []
        try:
            for idx, i in enumerate(sentence):
                w = self._idx_to_word[i]
                w_str = tf.compat.as_str(w)
                words.append(w_str)
            return " ".join(words)
            #return " ".join([tf.compat.as_str(self._idx_to_word[i]) for i in sentence])
        except UnicodeDecodeError  as e:
            print("Error: ", e)
            print("Final index:", idx, "and token:", i)
            print("Final word: ", self._idx_to_word[i])
            print("Sentence length:", len(sentence))
            print("\n\nIndexError encountered for following sentence:\n", sentence)
            print("\nVocab size is :", self.vocab_size)
            print("Words:", words)

    @property
    def name(self):
        """Returns name of the dataset as a string."""
        return self._name

    @property
    def train_size(self):
        raise NotImplemented

    @property
    def valid_size(self):
        raise NotImplemented

    @property
    def max_seq_len(self):
        return self._max_seq_len

    @staticmethod
    def fill_params(dataset_params):
        """Assigns default values from DEFAULT_FULL_CONFIG for keys not in dataset_params."""
        return {**DEFAULT_PARAMS, **dataset_params}
