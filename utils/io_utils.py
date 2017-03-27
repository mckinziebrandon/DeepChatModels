"""Utilities for downloading data from various datasets, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import yaml
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# Special vocabulary symbols.
_PAD = b"_PAD"      # Append to unused space for both encoder/decoder.
_GO  = b"_GO"       # Prepend to each decoder input.
_EOS = b"_EOS"      # Append to outputs only. Stopping signal when decoding.
_UNK = b"_UNK"      # For any symbols not in our vocabulary.
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# Enumerations for ease of use by this and other files.
PAD_ID  = 0
GO_ID   = 1
EOS_ID  = 2
UNK_ID  = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE   = re.compile(br"\d")

utils_dir = os.path.dirname(os.path.realpath(__file__))

def save_hyper_params(hyper_params, fname):
    # Append to file if exists, else create.
    df = pd.DataFrame(hyper_params)
    with open(fname, 'a+') as f:
        df.to_csv(f, header=False)


def get_sentence():
    """Simple function to prompt user for input and return it w/o newline.
    Frequently used in chat sessions, of course.
    """
    sys.stdout.write("Human: ")
    sys.stdout.flush()
    return sys.stdin.readline().strip().lower() # Could just use input() ...


def parse_config(config_path):
    """
    Args:
        config_path: (str) location of [my config].yml file.
               Both relative and absolute paths will work.

    Returns:
    """

    #config_path = os.path.abspath(config_path)
    config_path = os.path.join(utils_dir, '../configs', os.path.basename(config_path))
    with tf.gfile.GFile(config_path) as config_file:
        configs = yaml.load(config_file)
    return configs


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocabulary_path: path where the vocabulary will be created.
      data_path: data file that will be used to create vocabulary.
      max_vocabulary_size: limit on the size of the created vocabulary.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """

    if gfile.Exists(vocabulary_path): return

    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
        counter = 0
        for line in f:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)

            line   = tf.compat.as_bytes(line)
            tokens = basic_tokenizer(line)
            for w in tokens:
                word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        # Get sorted vocabulary, from most frequent to least frequent.
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = vocab_list[:max_vocabulary_size]

        # Write the list to a file.
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def get_vocab_dicts(vocabulary_path):
    """Returns word_to_idx, idx_to_word dictionaries given vocabulary.

    Args:
      vocabulary_path: path to the file containing the vocabulary.

    Returns:
      a pair: the vocabulary (a dictionary mapping string to integers), and
      the reversed vocabulary (a list, which reverses the vocabulary mapping).

    Raises:
      ValueError: if the provided vocabulary_path does not exist.
    """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.

    For example, a sentence "I have a dog" may become tokenized into
    ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
    "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

    Args:
      sentence: the sentence in bytes format to convert to token-ids.
      vocabulary: a dictionary mapping tokens to integers.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.

    Returns:
      a list of integers, the token-ids for the sentence.
    """
    words = basic_tokenizer(sentence)

    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]

    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data_path, calls the above
    sentence_to_token_ids, and saves the result to target_path. See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = get_vocab_dicts(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir, from_train_path, to_train_path,
                 from_dev_path, to_dev_path, from_vocabulary_size, to_vocabulary_size):
    """Prepare all necessary files that are required for the training.

      Args:
        data_dir: directory in which the data sets will be stored.
        from_train_path: path to the file that includes "from" training samples.
        to_train_path: path to the file that includes "to" training samples.
        from_dev_path: path to the file that includes "from" dev samples.
        to_dev_path: path to the file that includes "to" dev samples.
        from_vocabulary_size: size of the "from language" vocabulary to create and use.
        to_vocabulary_size: size of the "to language" vocabulary to create and use.

      Returns:
        A tuple of 6 elements:
          (1) path to the token-ids for "from language" training data-set,
          (2) path to the token-ids for "to language" training data-set,
          (3) path to the token-ids for "from language" development data-set,
          (4) path to the token-ids for "to language" development data-set,
          (5) path to the "from language" vocabulary file,
          (6) path to the "to language" vocabulary file.
      """
    # Create vocabularies of the appropriate sizes.
    to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
    from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
    create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size)
    create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size)

    # Create token ids for the training data.
    to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
    from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path)
    data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path)

    # Create token ids for the development data.
    to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
    from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
    data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path)
    data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path)

    train_ids_path = [from_train_ids_path, to_train_ids_path]
    dev_ids_path = [from_dev_ids_path, to_dev_ids_path]
    vocab_path = [from_vocab_path, to_vocab_path]
    return (train_ids_path, dev_ids_path, vocab_path)


def read_data(dataset, _buckets, max_train_data_size=None):
    """(NOT USED BY DYNAMIC MODELS) This is the main, and perhaps only,
    method that other files should use to access data.
    :return: train and validation sets of word IDS.
    """
    # Setup the data in appropriate directories and return desired PATHS.
    print("Preparing %s data in %s" % (dataset.name, dataset.data_dir))

    # Read data into buckets (e.g. len(train_set) == len(buckets)).
    train_set   = _read_data(dataset.paths['from_train'],
                             dataset.paths['to_train'],
                             _buckets, max_train_data_size)
    dev_set     = _read_data(dataset.paths['from_valid'],
                             dataset.paths['to_valid'], _buckets)
    return train_set, dev_set


def _read_data(source_path, target_path, _buckets, max_size=None):
    """(NOT USED BY DYNAMIC MODELS). Read data from source and target files,
        and put into buckets.

    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    # Counter for the number of source/target pairs that couldn't fit in _buckets.
    num_samples_too_large = 0
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                # Get source/target as list of word IDs.
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                # Place the source/target pair if they fit in a bucket.
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                    if bucket_id == len(_buckets) - 1:
                        num_samples_too_large += 1
                source, target = source_file.readline(), target_file.readline()

    print("Number of training samples that were too large for buckets:", num_samples_too_large)
    return data_set




