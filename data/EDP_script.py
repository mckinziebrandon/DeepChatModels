#!/usr/bin/env python3

"""
English Data Preprocessing script.
(converted from Jupyter notebook of same name in notebooks)
"""

import os
import re
import time
import json
import multiprocessing
from   functools        import wraps
from   itertools        import chain
from   collections      import Counter
from   multiprocessing  import Pool

import numpy  as np
import pandas as pd
from   progressbar import ProgressBar

# Some BS to emulate pyenchant's functionality (as far as we're concerned)
# to work around 64-bit Python installations on Windows.
try:
    from enchant import Dict
except ImportError:
    from nltk.corpus import wordnet
    def Dict(irrelevant_param):
        class Wrapper(object):
            @staticmethod
            def check(s):
                 return wordnet.synsets(word_to_test) == True
        return Wrapper

from data import DataHelper

# Global helper object that helps abstract away locations of
# files & directories, and keeps an eye on memory usage.
data_helper = DataHelper()
data_helper.file_counter = 6
# Max number of words in any saved sentence.
MAX_SEQ_LEN = 20
# Number of CPU cores available.
NUM_CORES = 1
# How many chunks we should split dataframes into at any given time.
NUM_PARTITIONS = 64


def timed_function(*expected_args):
    """Simple decorator to show how long the functions take to run."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start_time  = time.time()
            res         = fn(*args, **kwargs)
            stop_time   = time.time()
            fname = expected_args[0]
            print("Time to run %s: %.3f seconds." %
                  (fname, stop_time - start_time))
            return res
        return wrapper
    return decorator


@timed_function('parallel_map_df')
def parallel_map_df(fn, df):
    """Based on great explanation from 'Pandas in Parallel' (racketracer.com)."""
    df = np.array_split(df, NUM_PARTITIONS)
    pool = Pool(NUM_CORES)
    df   = pd.concat(pool.map(fn, df))
    pool.close()
    pool.join()
    return df


@timed_function('parallel_map_list')
def parallel_map_list(fn, iterable):
    """Based on great explanation from 'Pandas in Parallel' (racketracer.com)."""
    iterable = np.array_split(iterable, NUM_PARTITIONS)
    pool = Pool(NUM_CORES)
    iterable = np.concatenate(pool.map(fn, iterable))
    pool.close()
    pool.join()
    return iterable


def sentence_score(sentences):
    word_freq = data_helper.word_freq
    d = enchant.Dict('en_US')

    scores = []
    for sentence in sentences:
        word_count = len(sentence) + 1e-20
        sent_score = sum([1.0 / ((word_freq[w] + 1e-20) * word_count)
                      for w in sentence if not d.check(w)])
        scores.append(sent_score)
    return scores


def root_comments(df):
    '''Build list determining which rows of df are root comments.

    Returns:
        list of length equal to the number of rows in our data frame.
    '''
    root_value = []
    # Iterate over DataFrame rows as namedtuples,
    # with index value as first element of the tuple.
    for row in df.itertuples():
        root_value.append(row.parent_id == row.link_id)
    return root_value


@timed_function('remove_extra_columns')
def remove_extra_columns(df):
    """Throw away columns we don't need and misc. style formatting."""
    df['root'] = root_comments(df)
    df = df[['author', 'body', 'link_id', 'parent_id', 'name', 'root', 'subreddit']]
    df.style.set_properties(subset=['body'], **{'width': '500px'})
    df.style.set_properties(**{'text-align': 'left'})
    df.head()
    return df


@timed_function('regex_replacements')
def regex_replacements(df):
    # Remove comments that are '[deleted]'.
    df = df.loc[df.body != '[deleted]'].reset_index(drop=True)
    df.style.set_properties(subset=['body'], **{'width': '800px'})
    # Make all comments lowercase to help reduce vocab size.
    df['body'] = df['body'].map(lambda s: s.strip().lower())
    # Loop over regex replacements specified by modify_list.
    for old, new in data_helper.modify_list.items():
        df['body'].replace({old: new}, regex=True, inplace=True)
    # Remove comments with this extremely common occurrence.
    #df = df.loc[df.body != 'NUMBER'].reset_index(drop=True)
    return df


@timed_function('remove_large_comments')
def remove_large_comments(max_len, df):
    # Could probably do a regex find on spaces to make this faster.
    df = df[df['body'].map(lambda s: len(s.split())) < max_len].reset_index(drop=True)
    df = df[df['body'].map(lambda s: 'http' not in s)].reset_index(drop=True)
    return df


@timed_function('expand_contractions')
def expand_contractions(df):
    """Replace all contractions with their expanded form."""
    for contraction, as_words in data_helper.contractions.items():
        df['body'].replace({contraction: as_words}, regex=True, inplace=True)
    return df


@timed_function('children_dict')
def children_dict(df):
    """Returns a dictionary with keys being the root comments and
    values being their immediate root_to_children. Assumes that df has 'root' column.

    Go through all comments. If it is a root, skip it since they wont have a parent_id
    that corresponds to a comment.
    """
    children = {}
    for row in df.itertuples():
        if row.root == False:
            if row.parent_id in children.keys():
                children[row.parent_id].append(row.name)
            else:
                children[row.parent_id] = [row.name]
    return children


def main():

    # Get up to max_mem GiB of data.
    df = data_helper.safe_load()
    df = remove_extra_columns(df)
    df = regex_replacements(df)
    df = remove_large_comments(max_len=MAX_SEQ_LEN, df=df)
    df = expand_contractions(df)

    sentences = parallel_map_list(fn=DataHelper.word_tokenizer, iterable=df.body.values)
    data_helper.set_word_freq(Counter(chain.from_iterable(sentences)))

    print('Bout to score!')
    df['score'] = parallel_map_list(fn=sentence_score, iterable=sentences)
    sentences = None

    # Keep the desired percentage of lowest-scored sentences. (low score == good)
    keep_best_percent = 0.75
    df = df.loc[df['score'] < df['score'].quantile(keep_best_percent)]

    print('Prepping for the grand finale.')
    comments_dict       = pd.Series(df.body.values, index=df.name).to_dict()
    root_to_children    = children_dict(df)
    data_helper.generate_files(
        from_file_path="from_{}.txt".format(data_helper.file_counter),
        to_file_path="to_{}.txt".format(data_helper.file_counter),
        root_to_children=root_to_children,
        comments_dict=comments_dict)

if __name__ == '__main__':
    main()
