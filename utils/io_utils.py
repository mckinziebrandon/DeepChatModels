"""Utilities for downloading data from various datasets, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import yaml
import copy
import pandas as pd
import logging

import tensorflow as tf
from collections import Counter, namedtuple
from tensorflow.python.platform import gfile
from subprocess import Popen, PIPE
from chatbot.globals import DEFAULT_FULL_CONFIG


# Special vocabulary symbols.
_PAD = b"_PAD"      # Append to unused space for both encoder/decoder.
_GO = b"_GO"       # Prepend to each decoder input.
_EOS = b"_EOS"      # Append to outputs only. Stopping signal when decoding.
_UNK = b"_UNK"      # For any symbols not in our vocabulary.
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# Enumerations for ease of use by this and other files.
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# Build mock FLAGS object for utils to wrap info around if needed.
# This makes the API more user-friendly, since it takes care of
# formatting data if the user doesn't do it exactly as expected.
# Note: I did initially try this with an actual tf.app.flags object,
# but it was a nightmare.
_flag_names = ["pretrained_dir",
               "config",
               "debug",
               "model",
               "model_params",
               "dataset",
               "dataset_params"]
Flags = namedtuple('Flags', _flag_names)
_FLAGS = Flags(pretrained_dir=None,
               config=None,
               debug=None,
               model='{}',
               dataset='{}',
               model_params='{}',
               dataset_params='{}')


def save_hyper_params(hyper_params, fname):
    # Append to file if exists, else create.
    df = pd.DataFrame(hyper_params)
    with open(fname, 'a+') as f:
        df.to_csv(f, header=False)


def get_sentence(lower=True):
    """Simple function to prompt user for input and return it w/o newline.
    Frequently used in chat sessions, of course.
    """
    sys.stdout.write("Human: ")
    sys.stdout.flush()
    sentence = input()
    if lower:
        return sentence.lower()
    return sentence


def update_config(config=None,
                  config_path=None,
                  return_config=True,
                  **kwargs):
    """Update contents of a config file, overwriting any that 
    match those in kwargs.
   
    Args:
        config: (dict) subset of DEFAULT_FULL_CONFIG.
        config_path: (str) location of a yaml config file.
        return_config: (bool) whether or not to return the config dictionary.
        kwargs: key-value pairs to update in the config dictionary and/or file.
         
    At least one of {config, config_path} must be not None. If both are not
    None, then we update the config dictionary with the kwargs, and then set
    the file at config_path to match updated dictionary contents. 
    
    In other words, if config is not None, we do won't consider the contents 
    of config_path when doing the updates.
    """

    if config is None and config_path is None:
        raise ValueError("Configuration info not given to update_config.")

    if config is None:
        # Grab the current config file contents into a dictionary.
        config = get_yaml_config(config_path)
        logging.info("Updating config values %r for %s",
                     list(kwargs.keys()), config_path)

    # Update its values with those in kwargs.
    for top_level_key in DEFAULT_FULL_CONFIG:
        for update_key in kwargs:
            if update_key == top_level_key:
                config[update_key] = kwargs[update_key]
            elif update_key in DEFAULT_FULL_CONFIG[top_level_key]:
                if config.get(top_level_key) is None:
                    config[top_level_key] = {}
                config[top_level_key][update_key] = kwargs[update_key]

    # Rewrite the config file.
    if config_path is not None:
        with open(os.path.join(config_path), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    # Return the dictionary if requested.
    if return_config:
        return config


def get_yaml_config(path, save_path=True):
    with open(path) as file:
        config = yaml.load(file)
        if save_path:
            if config.get('dataset_params') is not None:
                config['dataset_params']['config_path'] = path
    return config


def load_pretrained_config(pretrained_dir):
    """Get the full configuration dictionary for a pretrained model.

    Args:
        pretrained_dir: path (relative to project root) that is assumed to contain:
        - config.yml: full configuration file (automatically saved by all models).
        - checkpoint(s) from training session (also saved automatically).

    Returns:
        config: dictionary loaded from config.yml, and with all training flags reset to
                chat session flags, since the only time this is called is for chatting.
    """
    config_path = os.path.join(pretrained_dir, "config.yml")
    config = get_yaml_config(config_path)
    # The loaded config will have "training" values, so we need
    # to set some of them to "chatting" values, instead of requiring
    # user to specify them (since they are mandatory for any chat sesion).
    config['model_params']['decode'] = True
    config['model_params']['is_chatting'] = True  # alias
    config['model_params']['reset_model'] = False
    config['model_params']['ckpt_dir'] = pretrained_dir
    return config


def print_non_defaults(config):
    """Prints all values in config that aren't the default values in DEFAULT_FULL_CONFIG.
    Args:
        config: dict of parameters with same structure as DEFAULT_FULL_CONFIG.
    """

    print("\n---------- Your non-default parameters: ----------")
    if config['model'] != DEFAULT_FULL_CONFIG['model']:
        print("{}: {}".format('model', config['model']))
    if config['dataset'] != DEFAULT_FULL_CONFIG['dataset']:
        print("{}: {}".format('dataset', config['dataset']))

    for dict_id in ['model_params', 'dataset_params']:
        print(dict_id, end=":\n")
        for key, val in config[dict_id].items():
            # First check if key isn't even specified by defaults.
            if key not in DEFAULT_FULL_CONFIG[dict_id]:
                print("\t{}: {}".format(key, val))
            elif DEFAULT_FULL_CONFIG[dict_id][key] != val:
                print("\t{}: {}".format(key, val))
    print("--------------------------------------------------\n")


def flags_to_dict(flags):
    """Builds and return a dictionary from flags keys, namely
       'model', 'dataset', 'model_params', 'dataset_params'.
    """

    if isinstance(flags, dict):
        logging.warning('The `flags` object is already a dictionary!')
        return flags

    if flags.pretrained_dir is not None:
        config = load_pretrained_config(flags.pretrained_dir)
        config['model_params'] = {**config['model_params'],
                                  **yaml.load(getattr(flags, 'model_params'))}
        return config

    flags_dict = {}
    # Grab any values under supported keys defined in default config.
    for stream in DEFAULT_FULL_CONFIG:

        stream_attr = getattr(flags, stream)
        if not isinstance(stream_attr, dict):
            yaml_stream = yaml.load(getattr(flags, stream))
        else:
            yaml_stream = stream_attr

        if yaml_stream:
            flags_dict.update({stream: yaml_stream})
        elif stream in ['model_params', 'dataset_params']:
            # Explicitly set it as empty for merging with default later.
            flags_dict[stream] = {}

    # If provided, incorporate yaml config file as well.
    # Give preference to values in flags_dict, since those are
    # values provided by user on command-line.
    if flags.config is not None:
        yaml_config = get_yaml_config(flags.config)
        flags_dict = merge_dicts(
            default_dict=yaml_config,
            preference_dict=flags_dict)

    return flags_dict


def merge_dicts(default_dict, preference_dict):
    """Preferentially (and recursively) merge input dictionaries.
    
    Ensures that all values in preference dict are used, and
    all other (i.e. unspecified) items are from default dict.
    """

    merged_dict = copy.deepcopy(default_dict)
    for pref_key in preference_dict:
        if isinstance(preference_dict[pref_key], dict) and pref_key in merged_dict:
            # Dictionaries are expected to have the same type structure.
            # So if any preference_dict[key] is a dict, then require default_dict[key]
            # must also be a dict (if it exists, that is).
            assert isinstance(merged_dict[pref_key], dict), \
                "Expected default_dict[%r]=%r to have type dict." % \
                (pref_key, merged_dict[pref_key])
            # Since these are both dictionaries, can just recurse.
            merged_dict[pref_key] = merge_dicts(merged_dict[pref_key],
                                                preference_dict[pref_key])
        else:
            merged_dict[pref_key] = preference_dict[pref_key]
    return merged_dict


def parse_config(flags=None, pretrained_dir=None, config_path=None):
    """Get custom configuration dictionary from either a tensorflow flags 
    object, a path to a training directory, or a path to a yaml file. Only pass 
    one of these. See "Args" below for more details.
    
    The result is a dictionary of the same key-val structure as seen in
    chatbot.globals.DEFAULT_FULL_CONFIG. For any key-value pair not found from
    the (single) argument passed, it will be set to the default found in
    DEFAULT_FULL_CONFIG.

    Args:
        flags: A tf.app.flags.FLAGS object. See FLAGS in main.py.
        pretrained_dir: relative [to project root] path to a pretrained model 
            directory, i.e. a directory where a chatbot was previously 
            saved/trained (a ckpt_dir).
        config_path: relative [to project root] path to a valid yaml 
            configuration file. For example: 'configs/my_config.yml'.

    Returns:
        config: dictionary of merged config info, where precedence is given to
        user-specified params on command-line (over .yml config files).
    """

    # Only pass one of the options!
    assert sum(x is not None for x in [flags, pretrained_dir, config_path]) == 1

    # Build a flags object from other params, if it doesn't exist.
    if flags is None:

        # Get the config_path from the pretrained directory.
        if config_path is None:
            config_path = os.path.join(pretrained_dir, 'config.yml')
        assert gfile.Exists(config_path), \
            "Cannot parse from %s. No config.yml." % config_path

        # Wrap flags string inside an actual tf.app.flags object.
        flags = _FLAGS
        flags = flags._replace(config=config_path)
    assert flags is not None

    # Get configuration dictionary containing user-specified parameters.
    config = flags_to_dict(flags)

    # Sanity check: make sure we have values that don't have defaults.
    if 'ckpt_dir' not in config['model_params']:
        print('Robot: Please enter a directory for saving checkpoints:')
        config['model_params']['ckpt_dir'] = get_sentence(lower=False)
    if 'data_dir' not in config['dataset_params']:
        print('Robot: Please enter full path to directory containing data:')
        config['dataset_params']['data_dir'] = get_sentence(lower=False)

    # Then, fill in any blanks with the full default config.
    config = merge_dicts(default_dict=DEFAULT_FULL_CONFIG,
                         preference_dict=config)
    return config


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().lower().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


def num_lines(file_path):
    """Return the number of lines in file given by its absolute path."""
    (num_samples, stderr) = Popen(['wc', '-l', file_path], stdout=PIPE).communicate()
    return int(num_samples.strip().split()[0])


def get_word_freqs(path, counter, norm_digits=True):
    """Extract word-frequency mapping from file given by path.
    
    Args:
        path: data file of words we wish to extract vocab counts from.
        counter: collections.Counter object for mapping word -> frequency.
        norm_digits: Boolean; if true, all digits are replaced by 0s.
    
    Returns:
        The counter (dict), updated with mappings from word -> frequency. 
    """

    print("Creating vocabulary for data", path)
    with gfile.GFile(path, mode="rb") as f:
        for i, line in enumerate(f):
            if (i + 1) % 100000 == 0:
                print("\tProcessing line", (i + 1))
            line = tf.compat.as_bytes(line)
            tokens = basic_tokenizer(line)
            # Update word frequency counts in vocab counter dict.
            for w in tokens:
                word = _DIGIT_RE.sub(b"0", w) if norm_digits else w
                counter[word] += 1
        return counter


def create_vocabulary(vocab_path, from_path, to_path, max_vocab_size, norm_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if norm_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocab_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
      vocab_path: path where the vocabulary will be created.
      from_path: data file for encoder inputs.
      to_path: data file for decoder inputs.
      max_vocab_size: limit on the size of the created vocabulary.
        norm_digits: Boolean; if true, all digits are replaced by 0s.
    """

    if gfile.Exists(vocab_path):
        return num_lines(vocab_path)

    vocab = Counter()
    # Pool all data words together to reflect the data distribution well.
    vocab = get_word_freqs(from_path, vocab, norm_digits)
    vocab = get_word_freqs(to_path, vocab, norm_digits)

    # Get sorted vocabulary, from most frequent to least frequent.
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    vocab_list = vocab_list[:max_vocab_size]

    # Write the list to a file.
    with gfile.GFile(vocab_path, mode="wb") as vocab_file:
        for w in vocab_list:
            vocab_file.write(w + b"\n")

    return len(vocab_list)


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
    sentence_to_token_ids, and saves the result to target_path.

    Args:
      data_path: path to the data file in one-sentence-per-line format.
      target_path: path where the file with token-ids will be created.
      vocabulary_path: path to the vocabulary file.
      normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = get_vocab_dicts(vocabulary_path=vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(
                        tf.compat.as_bytes(line), vocab, normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_data(data_dir,
                 vocab_size,
                 from_train_path=None,
                 to_train_path=None,
                 from_valid_path=None,
                 to_valid_path=None,
                 optimize=True,
                 config_path=None):

    """Prepare all necessary files that are required for the training.

    Args:
        data_dir: directory in which the data sets will be stored.
        from_train_path: path to the file that includes "from" training samples.
        to_train_path: path to the file that includes "to" training samples.
        from_valid_path: path to the file that includes "valid_from" samples.
        to_valid_path: path to the file that includes "valid_to" samples.
        vocab_size: preferred number of words to use in vocabulary.
        optimize: if True, allow program to rest this value if the actual
            vocab_size (num unique words in data) < preferred vocab_size. 
            This would decrease computational cost, should the situation arise.
        config_path: (required if optimize==True) location of config file.
        
    Note on optimize:
    - It will only have an effect if the following conditions are ALL met:
      - config_path is not None (and is a valid path)
      - optimize == True (of course)
      - true vocab size != [preferred] vocab_size

    Returns:
        Tuple of:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the vocabulary file,
        (6) the true vocabulary size (less than or equal to max allowed)
    """

    if optimize is None:
        logging.warning("You have not requested that your choice for "
                        "vocab_size be optimized. This can lead to slower "
                        "training times.\nSet 'optimize_params: true' under "
                        "dataset_params in your yaml config to enable.")

    def maybe_set_param(param, file_name):
        if param is None:
            param = os.path.join(data_dir, file_name)
            logging.info('Set path from None to %s', param)
        return param

    def get_vocab_path(vocab_size):
        return os.path.join(data_dir, "vocab%d.txt" % vocab_size)

    def append_to_paths(s, **paths):
        return {name: path + s for name, path in paths.items()}

    # Set any paths that are None to default values.
    from_train_path = maybe_set_param(from_train_path, 'train_from.txt')
    to_train_path = maybe_set_param(to_train_path, 'train_to.txt')
    from_valid_path = maybe_set_param(from_valid_path, 'valid_from.txt')
    to_valid_path = maybe_set_param(to_valid_path, 'valid_to.txt')

    # Create vocabularies of the appropriate sizes.
    vocab_path = get_vocab_path(vocab_size)
    true_vocab_size = create_vocabulary(
        vocab_path,
        from_train_path,
        to_train_path,
        vocab_size)
    assert true_vocab_size <= vocab_size

    # User-permitted, we reset the config file's vocab size and rename the
    # vocabulary path name to the optimal values.
    should_optimize = config_path is not None
    should_optimize = (vocab_size != true_vocab_size) and should_optimize
    should_optimize = optimize and should_optimize
    if should_optimize:
        logging.info('Optimizing vocab size in config and renaming files.')
        # Necessary when we overestimate the number of unique words in the data.
        # e.g. we set vocab_size = 40k but our data only has 5 unique words,
        # it would be wasteful to train a model on 40k.
        # Thus, we rename vocab filenames to have the true vocab size.
        vocab_size = true_vocab_size
        old_vocab_path = vocab_path
        vocab_path = get_vocab_path(true_vocab_size)
        if old_vocab_path != vocab_path:
            Popen(['mv', old_vocab_path, vocab_path], stdout=PIPE).communicate()

        # Reset the value of 'vocab_size' in the configuration file, so that
        # we won't need to regenerate everything again if the user wants to
        # resume training/chat/etc.
        update_config(config_path=config_path, vocab_size=true_vocab_size)

    id_paths = append_to_paths(
        '.ids%d' % vocab_size,
        from_train=from_train_path,
        to_train=to_train_path,
        from_valid=from_valid_path,
        to_valid=to_valid_path)

    # Create token ids for all training and validation data.
    for name in id_paths:
        data_to_token_ids(
            eval(name + '_path'),
            id_paths[name],
            vocab_path)

    return id_paths, vocab_path, vocab_size
