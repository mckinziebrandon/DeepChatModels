"""Because I'm tired of thinking about paths and filenames."""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from pprint import pprint
from pympler import asizeof # for profiling memory usage
import enchant
import json
from itertools import chain
from collections import Counter
from progressbar import ProgressBar

DATA_ROOTS = {'brandon': '/home/brandon/terabyte/Datasets/reddit',
              'ivan': '/Users/ivan/Documents/sp_17/reddit_data',
              'mitch': None}


class DataHelper:
    """Hi, I'm the DataHelper class. I manage file locations and how much computing resources
    are being used in the preprocessing stages. I make it harder for you to screw up."""

    def __init__(self, verbosity='info'):

        if verbosity == 'info':
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARN)

        self.log = logging.getLogger('DataHelperLogger')

        # Figure out who we are talking to: ivan or brandon.
        print("User name:")
        user = input().lower()
        if user not in ['brandon', 'ivan']: raise RuntimeError('Unknown User')
        data_root = DATA_ROOTS[user]
        print("Hello {}, I've set your data root as {}".format(user, data_root))

        # Which dataset we are working on.
        print("Dataset name: [reddit, ubuntu, wmt, cornell]")
        data_name = input().lower()
        if data_name == 'reddit':
            print("Year:")
            year = input()
            self.data_files = os.listdir(os.path.join(data_root, 'raw_data', year))
            print("These are the files I found for that year:")
            pprint(self.data_files)

        with open('dicts.json', 'r') as f:
            json_data = [json.loads(l) for l in f]
        # TODO: more descriptive names for the 'modify_' objects here would be nice.
        modify_list, modify_value, contractions = json_data

    def safe_load(self):
        """Load data while keeping an eye on memory usage."""

        # For in-place appending.
        # stackoverflow: import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
        list_ = []
        for i in range(len(self.data_files)):
            self.log.info("Starting to load file %s . . ." % self.data_files[i])
            # lines=True means "read as json-object-per-line."
            df = pd.read_json(self.data_files[0], lines=True)
            list_.append(df)
        df = pd.concat(df)

        init_num_rows = len(df.index)
        self.log.info("Number of lines in raw data file", init_num_rows)
        self.log.info("Column names from raw data file: %r" % df.columns)
        return df
