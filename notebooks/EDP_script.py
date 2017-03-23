import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from pprint import pprint
import enchant
import json
from itertools import chain
from collections import Counter
from progressbar import ProgressBar

with open('dicts.json', 'r') as f:
    json_data = [json.loads(l) for l in f]
modify_list, modify_value, contractions = json_data
modify_list = list(modify_list.items())

DATA_ROOT = '/home/brandon/terabyte/Datasets/reddit'
# Determine if this directory exists, if not use Ivan's directory.
if (os.path.isdir(DATA_ROOT)):
    pass
else:
    DATA_ROOT = '/Users/ivan/Documents/sp_17/reddit_data'
DATA_YEARS = ['2008']
# Use os.path.join; it will figure out the '/' in between.
RAW_DATA_FILES = [os.listdir(os.path.join(DATA_ROOT, 'raw_data', year)) for year in DATA_YEARS]

RAW_DATA_ABS_FILES = []
# Always work with full pathnames to be safe.
for i in range(len(DATA_YEARS)):
    for j in range(len(RAW_DATA_FILES[i])):
        if RAW_DATA_FILES[i][j].startswith('.'):
            pass
        else:
            RAW_DATA_ABS_FILES.append( os.path.join(DATA_ROOT, 'raw_data' , DATA_YEARS[i], RAW_DATA_FILES[i][j]))
RAW_DATA_FILES = RAW_DATA_ABS_FILES
pprint(RAW_DATA_FILES)

def load_data():
    print(RAW_DATA_FILES[0])
    df = pd.read_json(RAW_DATA_FILES[0], lines=True)
    for i in range(len(RAW_DATA_FILES) - 1):
        print("Read file", i)
        df = df.append(pd.read_json(RAW_DATA_FILES[i+1], lines=True), ignore_index=True)
    init_num_rows = len(df)
    print("Number of lines in raw data file", init_num_rows)
    pprint("Column names from raw data file:")
    pprint(df.columns)
    return df

def show_len_update(df):
    print("Now there are", len(df), "rows.")

def root_comments(df):
    '''Build list determining which rows of df are root comments.

    Returns:
        list of length equal to the number of rows in our data frame.
    '''
    root_value = []
    # Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple.
    for row in df.itertuples():
        root_value.append(row.parent_id == row.link_id)
    return root_value

def random_rows_generator(num_rows_per_print, num_rows_total):
    num_iterations = num_rows_total // num_rows_per_print
    shuffled_indices = np.arange(num_rows_per_print * num_iterations)
    np.random.shuffle(shuffled_indices)
    for batch in shuffled_indices.reshape(num_iterations, num_rows_per_print):
        yield batch

def initial_clean(df):
    df['root'] = root_comments(df)
    df = df[['author', 'body', 'link_id', 'parent_id', 'name', 'root', 'subreddit']]
    df.style.set_properties(subset=['body'], **{'width': '500px'})
    df.style.set_properties(**{'text-align': 'left'})
    show_len_update(df)
    df.head()
    return df

def clean_with_tracking(df):
    df = df.loc[df.body != '[deleted]'].reset_index(drop=True)
    df.style.set_properties(subset=['body'], **{'width': '800px'})
    df['body'] = df['body'].map(lambda s: s.strip().lower())

    total_mods = {}
    if 'mods' not in df:
        df['mods'] = np.zeros(len(df['body']), dtype=int)
    for patrn in modify_list:
        new_df = df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=False)
        modifications = list((np.where(new_df.values != df['body'].values))[0])
        df['body'] = new_df
        df['mods'][modifications] += modify_value[patrn[0]]
        total_mods[patrn[0]] = len(modifications)
    return df, total_mods

def remove_large_comments(n, df):
    print("Length before:", df['body'].size)
    df = df[df['body'].map(lambda s: len(s.split(' '))) < n].reset_index(drop=True)
    #show_len_update(df)
    return df

def load_data():
    print(RAW_DATA_FILES[0])
    df = pd.read_json(RAW_DATA_FILES[0], lines=True)
    for i in range(len(RAW_DATA_FILES) - 1):
        print("Read file", i)
        df = df.append(pd.read_json(RAW_DATA_FILES[i+1], lines=True), ignore_index=True)
    init_num_rows = len(df)
    print("Number of lines in raw data file", init_num_rows)
    pprint("Column names from raw data file:")
    pprint(df.columns)
    return df

def show_len_update(df):
    print("Now there are", len(df), "rows.")

def root_comments(df):
    '''Build list determining which rows of df are root comments.

    Returns:
        list of length equal to the number of rows in our data frame.
    '''
    root_value = []
    # Iterate over DataFrame rows as namedtuples, with index value as first element of the tuple.
    for row in df.itertuples():
        root_value.append(row.parent_id == row.link_id)
    return root_value

def random_rows_generator(num_rows_per_print, num_rows_total):
    num_iterations = num_rows_total // num_rows_per_print
    shuffled_indices = np.arange(num_rows_per_print * num_iterations)
    np.random.shuffle(shuffled_indices)
    for batch in shuffled_indices.reshape(num_iterations, num_rows_per_print):
        yield batch

def initial_clean(df):
    df['root'] = root_comments(df)
    df = df[['author', 'body', 'link_id', 'parent_id', 'name', 'root', 'subreddit']]
    df.style.set_properties(subset=['body'], **{'width': '500px'})
    df.style.set_properties(**{'text-align': 'left'})
    show_len_update(df)
    df.head()
    return df

def clean_with_tracking(df):
    df = df.loc[df.body != '[deleted]'].reset_index(drop=True)
    df.style.set_properties(subset=['body'], **{'width': '800px'})
    df['body'] = df['body'].map(lambda s: s.strip().lower())

    total_mods = {}
    if 'mods' not in df:
        df['mods'] = np.zeros(len(df['body']), dtype=int)
    for patrn in modify_list:
        new_df = df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=False)
        modifications = list((np.where(new_df.values != df['body'].values))[0])
        df['body'] = new_df
        df['mods'][modifications] += modify_value[patrn[0]]
        total_mods[patrn[0]] = len(modifications)
    return df, total_mods

def remove_large_comments(n, df):
    print("Length before:", df['body'].size)
    df = df[df['body'].map(lambda s: len(s.split(' '))) < n].reset_index(drop=True)
    #show_len_update(df)
    return df

import re
_WORD_SPLIT = re.compile(r'([.,!?\"\':;)(])|\s')
_DIGIT_RE   = re.compile(r"\d")

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = _WORD_SPLIT.split(sentence.strip())
    return [w for w in words if w]


def contraction_replacer(df):
    for patrn in contractions.items():
        df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=True)
    return df

def generate_files(from_file_path, to_file_path):
    """Generates two files, [from_file_path] and [to_file_path] of one-to-one comments
    """
    from_file_path = DATA_ROOT + '/' +  from_file_path
    to_file_path = DATA_ROOT + '/' + to_file_path
    ## Open the files and clear them.
    from_file = open(from_file_path, 'w')
    to_file = open(to_file_path, 'w')
    from_file.write("")
    to_file.write("")
    from_file.close()
    to_file.close()
    for key in children.keys():
        from_file = open(from_file_path, 'a')
        to_file = open(to_file_path, 'a')
        ## Since we have deleted comments, some comments parents might not exist anymore so we must catch that error.
        for child in children[key]:
            try:
                from_file.write(values_dict[key].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
                to_file.write(values_dict[child].replace('\n', '').replace('\r', ' ').replace('&gt', '') + "\n")
            except KeyError:
                pass
    from_file.close()
    to_file.close()

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

def reset():
    print('hi')
    df = load_data()
    print('hi2')
    df = initial_clean(df)
    print('hi3')
    df,total_mods = clean_with_tracking(df)
    df = remove_large_comments(10, df)
    print('removed comments with more than', 10, 'words.')
    df = contraction_replacer(df)
    print('I iz returnin')
    return df

import multiprocessing
print('startin')
df = reset()
print('yayz')

#sentences = [basic_tokenizer(sentence) for sentence in df['body']]
pool = multiprocessing.Pool()
sentences = list(pool.map(basic_tokenizer, df['body']))
words = [word for sentence in sentences for word in sentence]
word_freq = Counter(chain(words))
#word_freq = dict(word_freq)
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

print('sup')
pool = multiprocessing.Pool()
scores = list(pool.map(sentence_score, sentences))
df['score'] = [sum(s) for s in scores]
#add_sentence_scores(sentences)
print('done again')


df['score'] = [sum(s) for s in scores]
scores = None
print('k actually done')



df = df.loc[df.score < 0.008]
print('k sup')
values_dict = pd.Series(df.body.values, index=df.name).to_dict()
children = children_dict(df)
generate_files("from_file.txt", "to_file.txt")
print('writin this shit')
