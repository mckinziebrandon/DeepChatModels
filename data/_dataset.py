"""ABC for datasets. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import tensorflow as tf
from utils import io_utils
from abc import ABCMeta, abstractmethod, abstractproperty

from chatbot.globals import DEFAULT_FULL_CONFIG
DEFAULT_PARAMS = DEFAULT_FULL_CONFIG['dataset_params']


class DatasetABC(metaclass=ABCMeta):

    @abstractmethod
    def convert_to_tf_records(self, *args):
        """If not found in data dir, will create tfrecords data 
        files from text files.
        """
        pass

    @abstractmethod
    def train_generator(self, batch_size):
        """Returns a generator function for batches of batch_size 
        train data.
        """
        pass

    @abstractmethod
    def valid_generator(self, batch_size):
        """Returns a generator function for batches of batch_size 
        validation data.
        """
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
        """Return the maximum allowed sentence length."""
        pass


class Dataset(DatasetABC):

    def __init__(self, dataset_params):
        """Implements the general of subset of operations that all 
        dataset subclasses can use.

        Args:
            dataset_params: dictionary of configuration parameters. 
                See DEFAULT_FULL_CONFIG at top of file for supported keys.
        """

        self.__dict__['__params'] = Dataset.fill_params(dataset_params)
        # We query io_utils to ensure all data files are organized properly,
        # and io_utils returns the paths to files of interest.
        id_paths, vocab_path, vocab_size = io_utils.prepare_data(
            data_dir=self.data_dir,
            vocab_size=self.vocab_size,
            optimize=dataset_params.get('optimize_params'),
            config_path=dataset_params.get('config_path'))

        if vocab_size != self.vocab_size:
            self.log.info("Updating vocab size from %d to %d",
                          self.vocab_size, vocab_size)
            self.vocab_size = vocab_size
            # Also update the input dict, in case it is used later/elsewhere.
            dataset_params['vocab_size'] = self.vocab_size

        self.paths = dict()
        self.paths = {
            **id_paths,
            'vocab': vocab_path,
            'train_tfrecords': None,
            'valid_tfrecords': None}
        self._word_to_idx, self._idx_to_word = io_utils.get_vocab_dicts(
            vocab_path)

        # Create tfrecords file if not located in data_dir.
        self.convert_to_tf_records('train')
        self.convert_to_tf_records('valid')

    def convert_to_tf_records(self, prefix='train'):
        """If can't find tfrecords 'prefix' files, creates them.

        Args:
            prefix: 'train' or 'valid'. Determines which tfrecords to build.
        """

        from_path = self.paths['from_'+prefix]
        to_path = self.paths['to_'+prefix]
        tfrecords_fname = (prefix
                           + 'voc%d_seq%d' % (self.vocab_size, self.max_seq_len)
                           + '.tfrecords')
        output_path = os.path.join(self.data_dir, tfrecords_fname)
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
            decoder_list = [io_utils.GO_ID] \
                           + [int(x) for x in decoder_line.split()] \
                           + [io_utils.EOS_ID]

            # Why tensorflow . . . why . . .
            example.context.feature['encoder_sequence_length'].int64_list.value.append(
                len(encoder_list))
            example.context.feature['decoder_sequence_length'].int64_list.value.append(
                len(decoder_list))

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
                    encoder_line = encoder_file.readline()
                    decoder_line = decoder_file.readline()
                    while encoder_line and decoder_line:
                        sequence_example = get_sequence_example(
                            encoder_line,
                            decoder_line)
                        if sequence_example is not None:
                            writer.write(sequence_example.SerializeToString())
                        encoder_line = encoder_file.readline()
                        decoder_line = decoder_file.readline()

        self.log.info("Converted text files %s and %s into tfrecords file %s" \
                      % (os.path.basename(from_path),
                         os.path.basename(to_path),
                         os.path.basename(output_path)))
        self.paths[prefix + '_tfrecords'] = output_path

    def sentence_generator(self, prefix='from'):
        """Yields (as words) single sentences from training data, 
        for testing purposes.
        """
        self.log.info("Generating sentences from %s", self.paths[prefix+'_train'])
        with tf.gfile.GFile(self.paths[prefix+'_train'], mode="r") as f:
            sentence = self.as_words(
                list(map(int, f.readline().strip().lower().split())))
            while sentence:
                yield sentence
                sentence = self.as_words(
                    list(map(int, f.readline().strip().lower().split())))

    def pairs_generator(self, num_generate=None):
        in_sentences = self.sentence_generator('from')
        in_sentences = [s for s in in_sentences]
        out_sentences = self.sentence_generator('to')
        out_sentences = [s for s in out_sentences]

        if num_generate is None:
            num_generate = len(in_sentences)

        count = 0
        for in_sent, out_sent in zip(in_sentences, out_sentences):
            yield in_sent, out_sent

            count += 1
            if count >= num_generate:
                break

    def train_generator(self, batch_size):
        """[Note: not needed by DynamicBot since InputPipeline]"""
        return self._generator(
            self.paths['from_train'],
            self.paths['to_train'],
            batch_size)

    def valid_generator(self, batch_size):
        """[Note: not needed by DynamicBot since InputPipeline]"""
        return self._generator(
            self.paths['from_valid'],
            self.paths['to_valid'],
            batch_size)

    def _generator(self, from_path, to_path, batch_size):
        """(Used by BucketModels only). Returns a generator function that 
        reads data from file, and yields shuffled batches.

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
            encoder_batch = np.array(
                [s + [io_utils.PAD_ID] * (max_sent_len - len(s))
                 for s in encoder_tokens])[:, ::-1]
            decoder_batch = np.array(
                [s + [io_utils.PAD_ID] * (max_sent_len - len(s))
                 for s in decoder_tokens])
            return encoder_batch, decoder_batch

        encoder_tokens = []
        decoder_tokens = []
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:

                source, target = source_file.readline(), target_file.readline()
                while source and target:

                    # Skip sentence pairs that are too long for specifications.
                    space_needed = max(len(source.split()), len(target.split()))
                    if space_needed > self.max_seq_len:
                        source, target = source_file.readline(), target_file.readline()
                        continue

                    # Reformat token strings to token lists.
                    # Note: GO_ID is prepended by the chat bot, since it
                    # determines whether or not it's responsible for responding.
                    encoder_tokens.append([int(x) for x in source.split()])
                    decoder_tokens.append(
                        [int(x) for x in target.split()] + [io_utils.EOS_ID])

                    # Have we collected batch_size number of sentences?
                    # If so, pad & yield.
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
        """Convert list of integer tokens to a single sentence string."""

        words = []
        for token in sentence:
            word = self.idx_to_word[token]
            try:
                word = tf.compat.as_str(word)
            except UnicodeDecodeError:
                logging.error("UnicodeDecodeError on (token, word): "
                              "(%r, %r)", token, word)
                word = str(word)
            words.append(word)

        words = " ".join(words)
        #words = " ".join([tf.compat.as_str(self.idx_to_word[i]) for i in sentence])
        words = words.replace(' , ', ', ').replace(' .', '.').replace(' !', '!')
        words = words.replace(" ' ", "'").replace(" ?", "?")
        if len(words) < 2:
            return words
        return words[0].upper() + words[1:]

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
        """Assigns default values from DEFAULT_FULL_CONFIG 
        for keys not in dataset_params."""
        if 'data_dir' not in dataset_params:
            raise ValueError('data directory not found in dataset_params.')
        return {**DEFAULT_PARAMS, **dataset_params}

    def __getattr__(self, name):
        if name not in self.__dict__['__params']:
            raise AttributeError(name)
        else:
            return self.__dict__['__params'][name]

