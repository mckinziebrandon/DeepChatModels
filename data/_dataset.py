""" ABC for datasets. """
import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod, abstractproperty
from utils import io_utils
from utils.io_utils import EOS_ID, PAD_ID
import logging

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
    def data_dir(self):
        """Return path to directory that contains the data."""
        pass

    @abstractproperty
    def name(self):
        """Returns name of the dataset as a string."""
        pass

    @abstractproperty
    def train_size(self):
        """Returns number of samples (sentences) in this dataset."""
        pass

    @abstractproperty
    def valid_size(self):
        """Returns number of samples (sentences) in this dataset."""
        pass

    @abstractproperty
    def train_data(self):
        """List of training samples (token IDs)."""
        pass

    @abstractproperty
    def valid_data(self):
        """List of validation samples (token IDs)."""
        pass

    @abstractproperty
    def max_seq_len(self):
        """Return the number of tokens in the longest example"""
        pass


class Dataset(DatasetABC):

    def __init__(self, data_dir, vocab_size=20000, max_seq_len=80):
        """Implements the most general of subset of operations that all classes can use."""
        self.vocab_size = vocab_size
        self._data_dir = data_dir
        # We query io_utils to ensure all data files are organized properly,
        # and io_utils returns the paths to files of interest.
        paths_triplet = io_utils.prepare_data(self._data_dir,
                                              self._data_dir + "/train_from.txt",
                                              self._data_dir + "/train_to.txt",
                                              self._data_dir + "/valid_from.txt",
                                              self._data_dir + "/valid_to.txt",
                                              vocab_size, vocab_size)

        train_path, valid_path, vocab_path = paths_triplet
        self.paths = {}
        self.paths['from_train']    = train_path[0]
        self.paths['to_train']      = train_path[1]
        self.paths['from_valid']    = valid_path[0]
        self.paths['to_valid']      = valid_path[1]
        self.paths['from_vocab']    = vocab_path[0]
        self.paths['to_vocab']      = vocab_path[1]

        self._word_to_idx, _ = io_utils.get_vocab_dicts(self.paths['from_vocab'])
        _, self._idx_to_word = io_utils.get_vocab_dicts(self.paths['to_vocab'])

    def _generator(self, from_path, to_path, batch_size, max_seq_len=80):

        def padded_batch(sentences, max_length):
            padded = np.array([s + [PAD_ID] * (max_length - len(s)) for s in sentences])
            return padded

        def longest_sentence(enc_list, dec_list):
            max_enc_len = max([len(s) for s in enc_list])
            max_dec_len = max([len(s) for s in dec_list])
            return max(max_enc_len, max_dec_len)

        encoder_tokens = []
        decoder_tokens = []
        num_skipped = 0
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get the next sentence as integer token IDs.
                    if len(source.split()) > max_seq_len or len(target.split()) > max_seq_len:
                        source, target = source_file.readline(), target_file.readline()
                        num_skipped += 1
                        continue
                    encoder_tokens.append([int(x) for x in source.split()])
                    decoder_tokens.append([int(x) for x in target.split()] + [EOS_ID])
                    # Have we collected batch_size number of sentences? If so, pad & yield.
                    assert len(encoder_tokens) == len(decoder_tokens)
                    if len(encoder_tokens) == batch_size:
                        max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
                        # Encoder sentences are fed in reverse intentionally.
                        encoder_batch = padded_batch(encoder_tokens, max_sent_len)[:, ::-1]
                        decoder_batch = padded_batch(decoder_tokens, max_sent_len)
                        if num_skipped > 2:
                            print("Skipped %d sentences making batch." % num_skipped)
                        # Shuffling. Likely not that effective on batch by batch basis.
                        rand_indices = np.random.permutation(batch_size)
                        encoder_batch = encoder_batch[rand_indices]
                        decoder_batch = decoder_batch[rand_indices]
                        yield encoder_batch, decoder_batch
                        # Clear token containers for next batch.
                        encoder_tokens = []
                        decoder_tokens = []
                        num_skipped = 0
                    source, target = source_file.readline(), target_file.readline()
                # Don't forget to yield the 'leftovers'!
                assert len(encoder_tokens) == len(decoder_tokens)
                assert len(encoder_tokens) <= batch_size
                if len(encoder_tokens) > 0:
                    max_sent_len = longest_sentence(encoder_tokens, decoder_tokens)
                    encoder_batch = padded_batch(encoder_tokens, max_sent_len)[:, ::-1]
                    decoder_batch = padded_batch(decoder_tokens, max_sent_len)
                    yield encoder_batch, decoder_batch

    def _read_data(self, from_path, to_path):
        """(Deprecated, use generator methods instead).
        Read entire dataset into memory. Can be prohibitively memory intensive for large
        datasets.
        Args:
            suffix: (str) either "train" or "valid"

        Returns:
            2-tuple (source_data, target_data) of sentence-token lists.
        """
        source_data = []
        target_data = []
        # Counter for the number of source/target pairs that couldn't fit in _buckets.
        with tf.gfile.GFile(from_path, mode="r") as source_file:
            with tf.gfile.GFile(to_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get source/target as list of word IDs.
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(EOS_ID)
                    # Add to data_set and retrieve next id list.
                    source_data.append(source_ids)
                    target_data.append(target_ids)
                    source, target = source_file.readline(), target_file.readline()
        return source_data, target_data

    def train_generator(self, batch_size):
        """Returns a generator function. Each call to next() yields a batch
            of size batch_size data as numpy array of shape [batch_size, max_seq_len],
            where max_seq_len is the longest sentence in the returned batch.
        """
        return self._generator(self.paths['from_train'], self.paths['to_train'],
                               batch_size)

    def valid_generator(self, batch_size):
        return self._generator(self.paths['from_valid'], self.paths['to_valid'], batch_size)

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
    def data_dir(self):
        """Return path to directory that contains the data."""
        return self._data_dir

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
    def train_data(self):
        """Removed. Use generator instead."""
        self.log.error("Tried getting full training data. Use train_generator instead.")
        raise NotImplemented

    @property
    def valid_data(self):
        """Removed. Use generator instead."""
        self.log.error("Tried getting full validation data. Use valid_generator instead.")
        raise NotImplemented

    @property
    def max_seq_len(self):
        self.log.error("Tried getting max_seq_len. Unsupported since DynamicBot.")
        raise NotImplemented

    # =========================================================================================
    # Deprecated methods that have been replaced by better/more efficent ones.
    # Keeping in case users want to load full dataset at once/don't care about memory usage.
    # =========================================================================================

    def read_data(self, suffix="train"):
        return self._read_data(self.paths['from_%s' % suffix], self.paths['to_%s' % suffix])

