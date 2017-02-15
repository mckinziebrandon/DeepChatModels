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

