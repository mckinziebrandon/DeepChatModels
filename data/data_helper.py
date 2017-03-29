"""For use in preprocessing stages. Because I'm tired of thinking about
paths and filenames. Right now, is mainly for use by Brandon/Ivan. Will extend to
general users in the future.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from pprint import pprint
from pympler.asizeof import asizeof # for profiling memory usage
import json
from progressbar import ProgressBar

DATA_ROOTS = {'brandon': '/home/brandon/terabyte/Datasets/reddit',
              'ivan': '/Users/ivan/Documents/sp_17/reddit_data',
              'mitch': None}

HERE = os.path.dirname(os.path.realpath(__file__))


class DataHelper:
    """Hi, I'm the DataHelper class. I manage file locations and how much computing resources
    are being used in the preprocessing stages. I make it harder for you to screw up."""

    def __init__(self):

        self.log = logging.getLogger('DataHelperLogger')
        # Figure out who we are talking to: ivan or brandon.
        print("User name:", end=" ")
        user = input().lower()
        if user not in ['brandon', 'ivan']: raise RuntimeError('Unknown User')
        self.data_root = DATA_ROOTS[user]
        print("Hello {}, I've set your data root as {}".format(user, self.data_root))

        print("Year:", end=" ")
        year = input()
        data_files_rel_path = os.listdir(os.path.join(self.data_root, 'raw_data', year))
        self.data_files = [os.path.join(self.data_root, 'raw_data', year, fname)
                           for fname in data_files_rel_path]
        print("These are the files I found for that year:")
        pprint(self.data_files)

        with open(os.path.join(HERE, 'dicts.json'), 'r') as f:
            json_data = [json.loads(l) for l in f]
        # TODO: more descriptive names for the 'modify_' objects here would be nice.
        self.modify_list, self.modify_value, self.contractions = json_data

    def df_generator(self):
        """Generates df from single files at a time."""
        list_ = []
        for i in range(len(self.data_files)):
            df = pd.read_json(self.data_files[0], lines=True)
            init_num_rows = len(df.index)
            self.log.info("Number of lines in raw data file", init_num_rows)
            self.log.info("Column names from raw data file: %r" % df.columns)
            yield df

    def safe_load(self):
        """Load data while keeping an eye on memory usage."""

        # For in-place appending.
        # stackoverflow: import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
        list_ = []
        pbar = ProgressBar()
        for i in pbar(range(len(self.data_files))):
            self.log.info("Starting to load file %s . . ." % self.data_files[i])
            # lines=True means "read as json-object-per-line."
            df = pd.read_json(self.data_files[0], lines=True)
            list_.append(df)
            self.log.info("Data list has size %.3f GiB" % (float(asizeof(list_)) / 1e9))
        df = pd.concat(list_)

        init_num_rows = len(df.index)
        self.log.info("Number of lines in raw data file", init_num_rows)
        self.log.info("Column names from raw data file: %r" % df.columns)
        return df

    def generate_files(self, from_file_path, to_file_path, conversation_ids, data):
        """Generates two files, [from_file_path] and [to_file_path] of one-to-one comments
        """
        from_file_path = os.path.join(self.data_root, from_file_path)
        to_file_path = os.path.join(self.data_root, to_file_path)

        # Open the files and clear them.
        from_file   = open(from_file_path, 'w')
        to_file     = open(to_file_path, 'w')
        from_file.write("")
        to_file.write("")
        from_file.close()
        to_file.close()
        for context, responses in conversation_ids:
            from_file = open(from_file_path, 'a')
            to_file = open(to_file_path, 'a')
            # Since we have deleted comments, some comments parents might not exist anymore so we must catch that error.
            for resp in responses:
                try:
                    from_file.write(data[context].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
                    to_file.write(data[resp].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
                except KeyError:
                    pass
        from_file.close()
        to_file.close()
