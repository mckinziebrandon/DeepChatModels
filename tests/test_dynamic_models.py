"""Complete mock model creation, training, and decoding from start to finish."""

import numpy as np
import logging
import sys
import pdb
sys.path.append("..")
import tensorflow as tf
from utils import data_utils
from utils.data_utils import batch_concatenate
from utils import Dataset

class TestData(Dataset):
    """Mock dataset with a handful of sentences."""

    def __init__(self, vocab_size=100):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestDataLogger')
        self._name = "test_data"
        self.vocab_size = vocab_size
        self._data_dir = '/home/brandon/terabyte/Datasets/test_data'
        paths_triplet = data_utils.prepare_data(self._data_dir,
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

        if tf.gfile.Exists(self.paths['from_vocab']):
            rev_vocab = []
            with tf.gfile.GFile(self.paths['from_vocab'], mode="rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        else:
            raise ValueError("Vocabulary file %s not found.", self.paths['from_vocab'])

        self._word_to_idx = vocab
        self._idx_to_word = rev_vocab

    def word_to_idx(self):
        """Return dictionary map from str -> int. """
        return self._word_to_idx

    def idx_to_word(self):
        """Return dictionary map from int -> str. """
        return self._idx_to_word

    def translate(self, sentence):
        return " ".join([tf.compat.as_str(idx_to_word[i]) for i in source_ids]) + "."

    def data_dir(self):
        """Return path to directory that contains the data."""
        return self._data_dir

    def name(self):
        """Returns name of the dataset as a string."""
        return self._name

    def read_data(self, suffix="train"):
        data_set = []
        # Counter for the number of source/target pairs that couldn't fit in _buckets.
        with tf.gfile.GFile(self.paths['from_%s' % suffix], mode="r") as source_file:
            with tf.gfile.GFile(self.paths['to_%s' % suffix], mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                while source and target:
                    # Get source/target as list of word IDs.
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(data_utils.EOS_ID)
                    # Add to data_set and retrieve next id list.
                    data_set.append([source_ids, target_ids])
                    source, target = source_file.readline(), target_file.readline()
        return data_set



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('MainLogger')

    batch_size  = 2
    vocab_size  = 100
    state_size  = 128
    embed_size  = 64

    # Get the data in format of word ID lists.
    dataset = TestData(vocab_size)
    idx_to_word = dataset.idx_to_word()
    train_set = dataset.read_data("train")

    # ==============================================================================
    # Manual data preprocessing.
    # ==============================================================================

    # Just get a list of sentence ids for this simple test.
    sentences = []
    for source_ids, _ in train_set:
        sentences.append(source_ids)

    batch_sentences, sequence_lengths = batch_concatenate(sentences,
                                                          batch_size,
                                                          return_lengths=True)
    max_seq_len = sequence_lengths.max()

    # ==============================================================================
    # Embedding.
    # ==============================================================================

    def get_embedded_inputs(batch_concat_inputs):
        with tf.variable_scope("embedded_inputs_scope"):
            # 1. Embed the inputs.
            embeddings = tf.get_variable("embeddings", [vocab_size, embed_size])
            batch_embedded_inputs = tf.nn.embedding_lookup(embeddings, batch_concat_inputs)
            # NO WAY. THIS IS AWESOME!!!
            embedded_inputs = tf.unstack(batch_embedded_inputs)
            for embed_sentence in embedded_inputs:
                assert(isinstance(embed_sentence, tf.Tensor))
                assert(embed_sentence.shape == (batch_size, max_seq_len, embed_size))
            return embedded_inputs

    # 0. The raw integer sequences will be stored in a placeholder tensor.
    batch_concat_inputs = tf.placeholder(tf.int32, batch_sentences.shape)
    embedded_inputs = get_embedded_inputs(batch_concat_inputs)

    # ==============================================================================
    # DynamicRNN model.
    # ==============================================================================

    with tf.variable_scope("model"):
        model_inputs = tf.placeholder(tf.float32, [batch_size, max_seq_len, embed_size])
        seq_len_ph = tf.placeholder(tf.int32, [batch_size])
        cell = tf.contrib.rnn.GRUCell(num_units=state_size)
        outputs, state = tf.nn.dynamic_rnn(cell,
                                           model_inputs,
                                           sequence_length=seq_len_ph,
                                           dtype=tf.float32)

    init_op = tf.global_variables_initializer()

    # ==============================================================================
    # Execution.
    # ==============================================================================

    with tf.Session() as sess:

        sess.run(init_op)

        input_feed = {batch_concat_inputs.name: batch_sentences}
        embed_outputs = sess.run(fetches=embedded_inputs, feed_dict=input_feed)

        num_batches = batch_sentences.shape[0]
        for batch in range(num_batches):
            input_feed = {model_inputs.name: embed_outputs[batch],
                          seq_len_ph.name: sequence_lengths[batch]}
            pure_happiness = sess.run(fetches=outputs, feed_dict=input_feed)
            #         .-'"""""'-.
            #      .'           `.
            #     /   .      .    \
            #    :                 :
            #    |    _________    |
            #    :   \        /    :
            #     \   `.____.'    /
            #      `.           .'
            #        `-._____.-'
