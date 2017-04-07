"""For use in preprocessing stages. Because I'm tired of thinking about
paths and filenames. Right now, is mainly for use by Brandon/Ivan. Will extend to
general users in the future.
"""
import os
import pdb
import logging
import pandas as pd
from pprint import pprint
from pympler.asizeof import asizeof # for profiling memory usage
import json
from progressbar import ProgressBar

HERE = os.path.dirname(os.path.realpath(__file__))
DATA_ROOTS = {'brandon': '/home/brandon/Datasets/reddit',
        'ivan': '/Users/ivan/Documents/sp_17/reddit_data',
        'mitch': '/Users/Mitchell/Documents/Chatbot/RedditData'}
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
        print("Hi. I currently only support helping with the reddit dataset. "
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
        if not years: years = '2007,2008' # default years
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
            self.modify_list, self.contractions = json_data


        self._word_freq = None

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


        prev_id = -1
        with open(from_file_path, 'w') as from_file:
            with open(to_file_path, 'w') as to_file:
                for root_ID, child_IDs in root_to_children.items():
                    for child_ID in child_IDs:
                        try:
                            #from_file.write(comments_dict[root_ID].strip() + '\n')
                            #to_file.write(comments_dict[child_ID].strip() + '\n')
                            from_file.write(comments_dict[root_ID].replace('\n', '').replace('\r', '').replace('&gt', '') + "\n")
                            to_file.write(comments_dict[child_ID].replace('\n', '').replace('\r', '').replace('&gt', '') + "\n")
                        except KeyError:
                            pass
