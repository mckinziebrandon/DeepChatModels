# Trying to make it easier to load common datasets.
from keras.utils.data_utils import get_file
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer
import numpy as np
import nltk
import pandas as pd

unknown_token = 'UNKNOWN'
DATASETS = {
    'nietzsche': get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"),
    'ubuntu': '/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus/src/train.csv'
}

def get(dataset_name):
    """ documentation here hi"""
    try:
        data_path =  DATASETS[dataset_name]
    except KeyError as e:
        print("KeyError: Tried getting {} but does not exist.".format(dataset_name))
        raise
    return data_path

def get_text(dataset_name, lower=True, as_word_list=False):
    data_path   = get(dataset_name)
    text        =  open(data_path).read()
    if lower:
        text = text.lower()
    if as_word_list:
        text = text_to_word_sequence(text)
    return text

def get_train_data(dataset_name, vocab_size=1000):
    SEQ_LEN = 3
    N_STEPS = 2
    if dataset_name == 'nietzsche':
        text = get_text(dataset_name)
        text_tokenized = nltk.word_tokenize(text)
        unique_words = sorted(list(set(text_tokenized)))

        vocab_size = min(vocab_size, len(unique_words))

        vocab_freqs = nltk.FreqDist(text_tokenized).most_common(vocab_size - 1)
        vocab_words = [w for w, _ in vocab_freqs]
        vocab_words.append(unknown_token)

        text_tokenized = [w if w in vocab_words else unknown_token for w in text_tokenized]
        assert (len(vocab_words) == vocab_size)

        word_to_idx = {w: i for i, w in enumerate(vocab_words)}
        # Get full text document in format of indices into vocabulary.
        text_as_idx = [word_to_idx[w] for w in text_tokenized]

        X_train = np.array([[text_as_idx[i + t] for t in range(SEQ_LEN)]
                            for i in np.arange(len(text_tokenized) - SEQ_LEN - 1, step=N_STEPS)])
        y_train = np.zeros(shape=(X_train.shape[0], vocab_size))
        for i in np.arange(X_train.shape[0]):
            y_train[i, text_as_idx[N_STEPS * (i + 1)]] = 1
    else:
        raise RuntimeError

    return X_train, y_train

def get_ubuntu(vocab_size):
    # First, we need to load the data directly into a dataframe from the train.csv file.
    df_train = pd.read_csv(get('ubuntu'))
    # Remove all examples with label = 0. (why would i want to train on false examples?)
    df_train = df_train.loc[df_train['Label'] == 1.0]
    # Don't care about the pandas indices in the df, so remove them.
    df_train = df_train.reset_index(drop=True)
    # Get the df as a single text string.
    def df_to_string(df):
        """ Expects df to be 3 columns of form above. """
        # Remove the 'label' column since we are only interested in the text here.
        df_text = df.copy()
        del df_text['Label']
        text = df_text['Context'].str.cat(sep=' ') + ' ' + df_text['Utterance'].str.cat(sep=' ')
        return text
    text_train = df_to_string(df_train)
    print("Tokenizing data string....")
    tokens_train = nltk.word_tokenize(text_train)
    vocab_train = sorted(set(tokens_train)) # Sorted 'alphabetically', NOT frequency!!
    freq_dist = nltk.FreqDist(tokens_train)
    most_common = freq_dist.most_common(vocab_size - 1)
    print("Constructing dictionaries . . . ")
    word_to_index = {w: i for i, w in enumerate(np.array(most_common)[:, 0])}
    word_to_index[unknown_token] = len(word_to_index)
    index_to_word = {i: w for i, w in word_to_index.items()}
    assert(len(word_to_index) == vocab_size)

    def array_to_indices(arr):
        text_tokenized = [nltk.word_tokenize(sent) for sent in arr]
        text_tokenized = [[w if w in word_to_index else unknown_token for w in sent]
            for sent in text_tokenized]
        return np.array([[word_to_index[w] for w in sent] for sent in text_tokenized])

    print("Converting words to indices . . . ")
    context_as_indices = array_to_indices(df_train['Context'].values)
    utter_as_indices = array_to_indices(df_train['Utterance'].values)
    return (context_as_indices, utter_as_indices), (word_to_index, index_to_word)

