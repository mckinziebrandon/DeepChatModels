import os
import pandas as pd
import numpy as np
from pprint import pprint
import enchant
import re
from itertools import chain
from collections import Counter
from progressbar import ProgressBar
from constants import CONTRACTIONS, MODIFY_LIST, MODIFY_VALUE
from multiprocessing import Pool

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE   = re.compile(r"\d")

DATA_ROOT = '/home/brandon/terabyte/Datasets/reddit'
# Determine if this directory exists, if not use Ivan's directory.
if (os.path.isdir(DATA_ROOT)):
    pass
else:
    DATA_ROOT = '/Users/ivan/Documents/sp_17/reddit_data'

FROM_FILE = os.path.join(DATA_ROOT, 'processed_data', 'from_file.txt')
TO_FILE = os.path.join(DATA_ROOT, 'processed_data', 'to_file.txt')
DATA_YEARS = ['2007', '2008']
# Use os.path.join; it will figure out the '/' in between.
RAW_DATA_FILES = [os.listdir(os.path.join(DATA_ROOT, 'raw_data', year)) for year in DATA_YEARS]

RAW_DATA_ABS_FILES = []
# Always work with full pathnames to be safe.
for i in range(len(DATA_YEARS)):
    for j in range(len(RAW_DATA_FILES[i])):
        if RAW_DATA_FILES[i][j].startswith('.'):
            pass
        else:
            RAW_DATA_ABS_FILES.append( os.path.join(DATA_ROOT, 'raw_data' ,DATA_YEARS[i], RAW_DATA_FILES[i][j]))


def load_data():
    pprint(RAW_DATA_ABS_FILES)
    df = pd.read_json(RAW_DATA_ABS_FILES[0], lines=True)
    for i in range(len(RAW_DATA_ABS_FILES) - 1):
        df = df.append(pd.read_json(RAW_DATA_ABS_FILES[i + 1], lines=True), ignore_index=True)
        print("Finished reading in %s" % RAW_DATA_ABS_FILES[i+1])
    df = df.reset_index(drop=True)
    init_num_rows = len(df)
    print("Number of lines in raw data file", init_num_rows)
    pprint("Column names from raw data file:")
    pprint(df.columns)
    return df


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]


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
    for patrn in MODIFY_LIST:
        new_df = df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=False)
        modifications = list((np.where(new_df.values != df['body'].values))[0])
        df['body'] = new_df
        df['mods'][modifications] += MODIFY_VALUE[patrn[0]]
        total_mods[patrn[0]] = len(modifications)
    return df, total_mods

def contraction_replacer(df):
    for patrn in CONTRACTIONS.items():
        df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=True)
    return df

def invalid_word(df):
    '''Goes through the content and determines whether an invalid word is
    present.

    The data frame should provide a body field which will be inspected.
    '''
    d = enchant.Dict("en_US")
    valid_sentences = [True] * len(df)
    misspelled_words = {}

    for idx, sentence in enumerate(df['body'].values):
        for word in sentence:
            if not d.check(word):
                if word in misspelled_words:
                    misspelled_words[word] += 1
                else:
                    misspelled_words[word] = 1
                valid_sentences[idx] = False
    print("There are %i valid sentences out of %i." % (sum(valid_sentences), len(valid_sentences)))
    print("There are %i misspelled words." % len(misspelled_words))
    return valid_sentences, misspelled_words

def sentence_score(sentence):
    d = enchant.Dict('en_US')
    word_count = len(sentence)
    score = 0
    for word in sentence:
        if not d.check(word):
            try:
                score = score + 1.0/word_freq[word]
            except ZeroDivisionError:
                score = score + 1.0
    try:
        return score / word_count
    except ZeroDivisionError:
        return 1

def add_sentence_scores(df):
    scores = []
    pbar = ProgressBar()
    for sentence in pbar(sentences):
        scores.append(sentence_score(sentence))
    df['score'] = scores

def remove_large_comments(n, df):
    print("Length before:", df['body'].size)
    df = df[df['body'].map(lambda s: len(s.split(' '))) < n].reset_index(drop=True)
    show_len_update(df)
    return df

def children_dict(df):
    children = {}
    for row in df.itertuples():
        if row.root == False:
            if row.parent_id in children.keys():
                children[row.parent_id].append(row.name)
            else:
                children[row.parent_id] = [row.name]
    return children

## Generates two files, [from_file_path] and [to_file_path] of one-to-one comments.
def generate_files(from_file_path, to_file_path):
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


df = load_data()
print("All data files have been read.")
df = initial_clean(df)
print("Initial clean done.")
df,total_mods = clean_with_tracking(df)
print("Clean with tracking done.")
df = remove_large_comments(60, df)
print("Removal of large comments done.")
df = contraction_replacer(df)
print("Replacement of contractions done.")

print("Starting tokenization of sentences")
sentences = [basic_tokenizer(sentence) for sentence in df['body']]
print("Tokenization of sentences done with length: %d", len(sentences))
words = [word for sentence in sentences for word in sentence]
print("Word list created with length: %d" % len(words))
word_freq = Counter(chain(words))
print("Word frequency created")
add_sentence_scores(df)
print("Done Generating sentence scores")
df = df.loc[df.score < 0.005]
print("value dictionary generating")
values_dict = pd.Series(df.body.values, index=df.name).to_dict()
print("value dictionary generated")
children = children_dict(df)
print("Children dictionary generating")
print("Starting file generation")
generate_files(FROM_FILE, TO_FILE)
