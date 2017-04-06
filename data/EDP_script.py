import os
import inspect
import pandas as pd
import numpy as np
from functools import wraps
from pprint import pprint
import time
import enchant
import json
from itertools import chain
from collections import Counter
from progressbar import ProgressBar
import multiprocessing
from data import DataHelper
import re
_WORD_SPLIT = re.compile(r'([.,!?\"\':;)(])|\s')
_DIGIT_RE   = re.compile(r"\d")

# Global helper object that helps abstract away locations of
# files & directories, and keeps an eye on memory usage.
data_helper = DataHelper()
# Max number of words in any saved sentence.
MAX_SEQ_LEN = 10

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

def root_comments(df):
    '''Build list determining which rows of df are root comments.

    Returns:
        list of length equal to the number of rows in our data frame.
    '''
    root_value = []
    # Iterate over DataFrame rows as namedtuples, with index value as
    # first element of the tuple.
    for row in df.itertuples():
        root_value.append(row.parent_id == row.link_id)
    return root_value


def random_rows_generator(num_rows_per_print, num_rows_total):
    num_iterations = num_rows_total // num_rows_per_print
    shuffled_indices = np.arange(num_rows_per_print * num_iterations)
    np.random.shuffle(shuffled_indices)
    for batch in shuffled_indices.reshape(num_iterations, num_rows_per_print):
        yield batch


@timed_function('initial_clean')
def initial_clean(df):
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
    return df


@timed_function('remove_large_comments')
def remove_large_comments(max_len, df):
    # Could probably do a regex find on spaces to make this faster.
    df = df[df['body'].map(lambda s: len(s.split(' '))) < max_len].reset_index(drop=True)
    return df


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = _WORD_SPLIT.split(sentence.strip())
    return [w for w in words if w]


@timed_function('expand_contractions')
def expand_contractions(df):
    """Replace all contractions with their expanded form."""
    for contraction, as_words in data_helper.contractions.items():
        df['body'].replace({contraction: as_words}, regex=True, inplace=True)
    return df


def children_dict(df):
    """Returns a dictionary with keys being the root comments and values being their immediate root_to_children.
        Assumes to have a 'root' column already.
        Go through all comments, if it is a root skip it since they wont have a parent_id corresponding
        to a comment.
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
    df = data_helper.safe_load(max_mem=1.0)
    df = initial_clean(df)
    df = regex_replacements(df)
    df = remove_large_comments(max_len=MAX_SEQ_LEN, df=df)
    df = expand_contractions(df)

    print('bye')
    exit()
    pool = multiprocessing.Pool()
    sentences = list(pool.map(basic_tokenizer, df['body']))
    words = [word for sentence in sentences for word in sentence]
    word_freq = Counter(chain(words))
    words = None

    def sentence_score(sentence):
        d = enchant.Dict('en_US')
        word_count = len(sentence)+1e-20
        sent_score = [1.0/((word_freq[w]+1e-20)*word_count) for w in sentence if not d.check(w)]
        return sent_score

    def add_sentence_scores(sentences):
        scores = []
        pbar = ProgressBar()
        i = 0
        for sentence in pbar(sentences):
            scores.append(sentence_score(sentence))
        df['score'] = scores

    print('Bout to score!')
    pool    = multiprocessing.Pool()
    scores  = list(pool.map(sentence_score, sentences))
    df['score'] = [sum(s) for s in scores]
    scores  = None
    df      = df.loc[df.score < 0.008]

    print('Prepping for the grand finale.')
    comments_dict       = pd.Series(df.body.values, index=df.name).to_dict()
    root_to_children    = children_dict(df)
    data_helper.generate_files(
        "from_file.txt", "to_file.txt", root_to_children, comments_dict
    )


if __name__ == '__main__':
    main()
