"""Nothing to see here . . . """

import os
import re
import numpy as np
import tensorflow as tf

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

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


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




def load_graph(frozen_model_dir):
    """Load frozen tensorflow graph into the default graph.

    Args:
        frozen_model_dir: location of protobuf file containing frozen graph.

    Returns:
        tf.Graph object imported from frozen_model_path.
    """

    # Prase the frozen graph definition into a GraphDef object.
    frozen_file = os.path.join(frozen_model_dir, "frozen_model.pb")
    with tf.gfile.GFile(frozen_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Load the graph def into the default graph and return it.
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None
        )
    return graph


def unfreeze_bot(frozen_model_path):
    """Restores the frozen graph from file and grabs input/output tensors needed to
    interface with a bot for conversation.

    Args:
        frozen_model_path: location of protobuf file containing frozen graph.

    Returns:
        outputs: tensor that can be run in a session.
    """

    bot_graph   = load_graph(frozen_model_path)
    tensors = {'inputs': bot_graph.get_tensor_by_name('import/input_pipeline/user_input:0'),
               'outputs': bot_graph.get_tensor_by_name('import/outputs:0')}
    return tensors, bot_graph

class FrozenBot:

    def __init__(self, frozen_model_dir, vocab_size):
        print(frozen_model_dir)
        print(type(frozen_model_dir))
        self.tensor_dict, self.graph = unfreeze_bot(frozen_model_dir)
        self.sess = tf.Session(graph=self.graph)

        self.config = {'dataset_params': {
            'data_dir': frozen_model_dir, 'vocab_size': vocab_size}}
        self.word_to_idx, self.idx_to_word = self.get_frozen_vocab()

    def get_frozen_vocab(self):
        """Helper function to get dictionaries for translating between tokens and words."""
        data_dir    = self.config['dataset_params']['data_dir']
        vocab_size  = self.config['dataset_params']['vocab_size']
        vocab_paths = {
            'from_vocab': os.path.join(data_dir, 'vocab{}.from'.format(vocab_size)),
            'to_vocab': os.path.join(data_dir, 'vocab{}.to'.format(vocab_size))}
        word_to_idx, _ = self.get_vocab_dicts(vocabulary_path=vocab_paths['from_vocab'])
        _, idx_to_word = self.get_vocab_dicts(vocabulary_path=vocab_paths['to_vocab'])
        return word_to_idx, idx_to_word

    def as_words(self, sentence):
        words = " ".join([tf.compat.as_str(self.idx_to_word[i]) for i in sentence])
        words = words.replace(' , ', ', ').replace(' .', '.').replace(' !', '!')
        return words[0].upper() + words[1:]

    def __call__(self, sentence):
        """Outputs response sentence (string) given input (string)."""
        # Convert input sentence to token-ids.
        sentence_tokens = sentence_to_token_ids(
            tf.compat.as_bytes(sentence), self.word_to_idx)
        sentence_tokens = np.array([sentence_tokens[::-1]])

        # Get output sentence from the chatbot.
        fetches = self.tensor_dict['outputs']
        feed_dict={self.tensor_dict['inputs']: sentence_tokens}
        response = self.sess.run(fetches=fetches, feed_dict=feed_dict)
        return self.as_words(response[0][:-1])

    def get_vocab_dicts(self, vocabulary_path):
        """Returns word_to_idx, idx_to_word dictionaries given vocabulary.

        Args:
          vocabulary_path: path to the file containing the vocabulary.

        Returns:
          a pair: the vocabulary (a dictionary mapping string to integers), and
          the reversed vocabulary (a list, which reverses the vocabulary mapping).

        Raises:
          ValueError: if the provided vocabulary_path does not exist.
        """
        if tf.gfile.Exists(vocabulary_path):
            rev_vocab = []
            with tf.gfile.GFile(vocabulary_path, mode="rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)


