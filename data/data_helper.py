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

HERE = os.path.dirname(os.path.realpath(__file__))
DATA_ROOTS = {'brandon': '/home/brandon/Datasets/reddit',
              'ivan': '/Users/ivan/Documents/sp_17/reddit_data'}
# Maximum memory usage allowed in GiB.
MAX_MEM = 2.0


class DataHelper:
    """Hi, I'm the DataHelper class. I manage file locations and how much computing resources
    are being used in the preprocessing stages. I make it harder for you to screw up."""

    def __init__(self):
        """Short convo with user that gives us what we need to help out."""

        # Keeps track of current file we're processing.
        self.file_counter = 0

        self.log = logging.getLogger('DataHelperLogger')
        print("Hi. I'm new to this. "
              "I currently only support helping with the reddit dataset. "
              "If you're working with another dataset, I'm sorry.")

        # 1. Get user name. We can associate info with a given user as we go.
        print("User name:", end=" ")
        user = input().lower()
        if not user:
            user = 'brandon'
        if user not in DATA_ROOTS:
            print("I don't recognize you, %s." % user)
            print("Please give me the path to your data:", end=" ")
            self.data_root = input()
        else:
            self.data_root = DATA_ROOTS[user]
        print("Hello {}, I've set your data root as {}".format(user, self.data_root))

        # 2. Get absolute paths to all data filenames in self.file_paths.
        self.file_paths = []
        print("Years to process (comma-separated):", end=" ")
        years = input()
        if not years: years = '2007,2008'
        years = years.split(',')
        for y in years:
            rel_paths = os.listdir(os.path.join(self.data_root, 'raw_data', y))
            self.file_paths.extend(
                [os.path.join(self.data_root, 'raw_data', y, f) for f in rel_paths]
            )
        print("These are the files I found:")
        pprint(self.file_paths)
        print()

        # Load the helper dictionaries from dicts.json.
        with open(os.path.join(HERE, 'dicts.json'), 'r') as f:
            json_data = [json.loads(l) for l in f]
            # TODO: more descriptive names for the 'modify_' objects here would be nice.
            self.modify_list, self.modify_value, self.contractions = json_data

    def df_generator(self):
        """Generates df from single files at a time."""
        for i in range(len(self.file_paths)):
            df = pd.read_json(self.file_paths[i], lines=True)
            init_num_rows = len(df.index)
            self.log.info("Number of lines in raw data file", init_num_rows)
            self.log.info("Column names from raw data file: %r" % df.columns)
            yield df

    def safe_load(self, max_mem=MAX_MEM):
        """Load data while keeping an eye on memory usage."""

        if self.file_counter >= len(self.file_paths):
            print("No more files to load!")
            return None

        # For in-place appending.
        # stackoverflow: import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
        list_ = []
        pbar = ProgressBar()
        for i in pbar(range(self.file_counter, len(self.file_paths))):

            self.log.info("Starting to load file %s . . ." % self.file_paths[i])
            # lines=True means "read as json-object-per-line."
            df = pd.read_json(self.file_paths[0], lines=True)
            list_.append(df)

            mem_usage = float(asizeof(list_)) / 1e9
            self.log.info("Data list has size %.3f GiB" % mem_usage)
            if mem_usage > max_mem:
                self.log.warning("At max capacity. Leaving data collection early.")
                break
        self.file_counter = i + 1

        df = pd.concat(list_)
        df = df.reset_index()
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
            for resp in responses:
                try:
                    from_file.write(data[context].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
                    to_file.write(data[resp].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
                except KeyError:
                    pass
        from_file.close()
        to_file.close()
