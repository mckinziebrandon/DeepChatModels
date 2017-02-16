# Trying to make it easier to load common datasets.
from keras.utils.data_utils import get_file
from keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer

DATASETS = {
    'nietzsche': get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
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
    import nltk
    import numpy as np
    SEQ_LEN = 3
    N_STEPS = 2
    if dataset_name == 'nietzsche':
        unknown_token = 'UNKNOWN'
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


