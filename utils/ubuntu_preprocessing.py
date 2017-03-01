import pdb
import numpy as np
import os.path
import pandas as pd
from pprint import pprint

DATA_DIR = '/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus/'
TRAIN_PATH = DATA_DIR + 'src/train.csv'
VALID_PATH = DATA_DIR + 'src/valid.csv'
TEST_PATH = DATA_DIR + 'src/test.csv'


def get_validation():
    # First, we need to load the data directly into a dataframe from the train.csv file.
    df_valid = pd.read_csv(VALID_PATH)
    first_two_cols = df_valid.columns[:2]
    df_valid = df_valid[first_two_cols]
    df_valid.columns = ['Context', 'Utterance']
    return df_valid

def get_training():
    # First, we need to load the data directly into a dataframe from the train.csv file.
    df_train = pd.read_csv(TRAIN_PATH)
    # Remove all examples with label = 0. (why would i want to train on false examples?)
    df_train = df_train.loc[df_train['Label'] == 1.0]
    # Don't care about the pandas indices in the df, so remove them.
    df_train = df_train.reset_index(drop=True)

    return df_train


def print_conversation(df, index=0):
    context_entry = df['Context'].values[index]
    turns = context_entry.split('__eot__')[:-1]

    print('--------------------- CONTEXT ------------------- ')
    for idx_turn, turn in enumerate(turns):

        utters_this_turn = turn.split('__eou__')[:-1]
        print("\nTurn {}: ".format(idx_turn))

        for idx_utter, utterance in enumerate(utters_this_turn):
            print("\tUtterance {}: ".format(idx_utter % 2), utterance)

    target = df['Utterance'].values[index]
    print('--------------------- RESPONSE ------------------- ')
    for idx_utter, utter in enumerate(target.split('__eou__')[:-1]):
        print("Utterance {}: ".format(idx_utter), utter)

def save_to_file(fname, arr):
    with open(DATA_DIR+fname,"w") as f:
        for line in arr:
            f.write(line + "\n")


def get_inputs(df):
    """
    col == "Context" or "Utterance"
    """
    encodes = []
    decodes = []
    for row in df["Context"].values:  # [:N_SAMPLES]:
        turns = row.split('__eot__')  # [:-1] #[-1]
        if len(turns) < 2: continue
        turns = turns[:-1]

        userOne = turns[0::2]
        userTwo = turns[1::2]

        if len(userOne) != len(userTwo):
            min_len = min(len(userOne), len(userTwo))
            userOne = userOne[:min_len]
            userTwo = userTwo[:min_len]

        userOne = ["".join(t.split('__eou__')[:-1]) for t in userOne]
        userTwo = ["".join(t.split('__eou__')[:-1]) for t in userTwo]

        encodes += userOne
        decodes += userTwo
    return encodes, decodes

def print_n_pairs(n, inputs, outputs):
    for i in range(n):
        print(inputs[i])
        print(outputs[i])
        print()


if __name__ == '__main__':
    df_train = get_training()
    df_valid = get_validation()


