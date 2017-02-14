# Trying to make it easier to load common datasets.
from keras.utils.data_utils import get_file

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

def get_text(dataset_name):
    """ yooo
    """

    data_path = get(dataset_name)
    return open(data_path).read()
