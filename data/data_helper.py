"""For use in preprocessing stages. Because I'm tired of thinking about
paths and filenames. Right now, is mainly for use by Brandon/Ivan. Will extend to
general users in the future.
"""
import os
import pdb
import re
import logging
import pandas as pd
import numpy as np
from pprint import pprint
from pympler.asizeof import asizeof # for profiling memory usage
import json
from progressbar import ProgressBar
from subprocess import Popen, PIPE

# Absolute path to this file.
_WORD_SPLIT = re.compile(r'([.,!?\"\':;)(])|\s')
HERE = os.path.dirname(os.path.realpath(__file__))
DATA_ROOTS = {'brandon': '/home/brandon/Datasets/reddit',
        'ivan': '/Users/ivan/Documents/sp_17/reddit_data',
        'mitch': '/Users/Mitchell/Documents/Chatbot/RedditData'}
# Maximum memory usage allowed in GiB.
MAX_MEM = 2.0



class DataHelper:
    """Hi, I'm the DataHelper class. I manage file locations and how much computing resources
    are being used in the preprocessing stages. I make it harder for you to screw up."""

    def __init__(self, log_level=logging.WARNING):
        """Short convo with user that gives us what we need to help out."""

        logging.basicConfig(filename='/tmp/data_helper.log', level=log_level)
        # Keeps track of current file we're processing.
        self.file_counter = 0
        # Temporary work-around for parallel-processing with frequency dict.
        self._word_freq = None

        print("Hi, I'm a DataHelper. For now, I support helping with the reddit dataset.")
        print("For all questions, simply press ENTER if you want the default value.")

        # 1. Get user name. We can associate info with a given user as we go.
        print("User name: (default=brandon)", end=" ")
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
        print("Years to process as comma-separated list: (default=2007,2008,2009)", end=" ")
        years = input()
        if not years: years = '2007,2008,2009'
        years = years.split(',')
        for y in years:
            rel_paths = os.listdir(os.path.join(self.data_root, 'raw_data', y))
            self.file_paths.extend(
                    [os.path.join(self.data_root, 'raw_data', y, f) for f in rel_paths])
            print("These are the files I found:")
        pprint(self.file_paths)
        print()

        print("Max memory to use? (default=%.2f GiB)" % MAX_MEM)
        _max_mem = input()
        if not _max_mem: self.max_mem = MAX_MEM
        else: self.max_mem = float(_max_mem)

        # Load the helper dictionaries from dicts.json.
        with open(os.path.join(HERE, 'dicts.json'), 'r') as f:
            json_data = [json.loads(l) for l in f]
            # TODO: more descriptive names for the 'modify_' objects here would be nice.
            self.modify_list, self.modify_value, self.contractions = json_data

    def safe_load(self):
        """Load data while keeping an eye on memory usage."""

        if self.file_counter >= len(self.file_paths):
            print("No more files to load!")
            return None

        # For in-place appending.
        # stackoverflow: import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
        list_ = []
        pbar = ProgressBar()
        for i in pbar(range(self.file_counter, len(self.file_paths))):

            print("Starting to load file %s . . ." % self.file_paths[i])
            # lines=True means "read as json-object-per-line."
            list_.append(pd.read_json(self.file_paths[i], lines=True))

            mem_usage = float(asizeof(list_)) / 1e9
            logging.info("Data list has size %.3f GiB" % mem_usage)
            if mem_usage > self.max_mem:
                print("At max capacity. Leaving data collection early.")
                break
        self.file_counter = i + 1

        df = pd.concat(list_).reset_index()
        init_num_rows = len(df.index)
        logging.info("Number of lines in raw data file", init_num_rows)
        logging.info("Column names from raw data file: %r" % df.columns)
        logging.info("DataHelper.safe_load: df.head() = %r" % df.head())
        return df

    def set_word_freq(self, wf):
        self._word_freq = wf

    @property
    def word_freq(self):
        return self._word_freq

    def generate_files(self, from_file_path, to_file_path, root_to_children, comments_dict):
        """Generates two files, [from_file_path] and [to_file_path] of one-to-one comments
        """
        from_file_path = os.path.join(self.data_root, from_file_path)
        to_file_path = os.path.join(self.data_root, to_file_path)

        with open(from_file_path, 'w') as from_file:
            with open(to_file_path, 'w') as to_file:
                for root_ID, child_IDs in root_to_children.items():
                    for child_ID in child_IDs:
                        try:
                            from_file.write(comments_dict[root_ID].strip() + '\n')
                            to_file.write(comments_dict[child_ID].strip() + '\n')
                        except KeyError:
                            pass

        (num_samples, stderr) = Popen(['wc', '-l', from_file_path], stdout=PIPE).communicate()
        num_samples = int(num_samples.strip().split()[0])
        print("Final processed file has %d samples total." % num_samples)
        # First make sure user has copy of bash script we're about to use.
        os.popen('cp %s %s' % (os.path.join(HERE, 'split_into_n.sh'), self.data_root))
        # Split data into 90% training and 10% validation.
        os.popen('bash %s %d' % (os.path.join(self.data_root, 'split_into_n.sh'),
                            0.1 * num_samples))



    def df_generator(self):
        """Generates df from single files at a time."""
        for i in range(len(self.file_paths)):
            df = pd.read_json(self.file_paths[i], lines=True)
            init_num_rows = len(df.index)
            logging.info("Number of lines in raw data file", init_num_rows)
            logging.info("Column names from raw data file: %r" % df.columns)
            yield df

    @staticmethod
    def random_rows_generator(num_rows_per_print, num_rows_total):
        """Fun generator for viewing random comments (rows) in dataframes."""
        num_iterations = num_rows_total // num_rows_per_print
        shuffled_indices = np.arange(num_rows_per_print * num_iterations)
        np.random.shuffle(shuffled_indices)
        for batch in shuffled_indices.reshape(num_iterations, num_rows_per_print):
            yield batch

    @staticmethod
    def word_tokenizer(sentences):
        """Tokenizes sentence/list of sentences into word tokens."""

        # Suggestion by user: Shaptic.
        tokenized = [None for _ in range(len(sentences))]
        for i in range(len(sentences)):
            tokenized[i] = [w for w in _WORD_SPLIT.split(sentences[i].strip()) if w]
        return tokenized

