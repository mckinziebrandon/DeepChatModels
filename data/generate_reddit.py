import os
import pandas as pd
import numpy as np
from pprint import pprint
import enchant
import re
from itertools import chain
from collections import Counter
from progressbar import ProgressBar

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
DATA_YEARS = ['2007']#, '2008']
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


CONTRACTIONS = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he shall",
        "he'll've": "he shall have",
        "he's": "he has",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has",
        "I'd": "I had",
        "I'd've": "I would have",
        "I'll": "I shall",
        "I'll've": "I shall have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it shall",
        "it'll've": "it shall have",
        "it's": "it has",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she shall",
        "she'll've": "she shall have",
        "she's": "she has",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that has",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there has",
        "they'd": "they had",
        "they'd've": "they would have",
        "they'll": "they shall",
        "they'll've": "they shall have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall",
        "what'll've": "what shall have",
        "what're": "what are",
        "what's": "what has",
        "what've": "what have",
        "when's": "when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has",
        "where've": "where have",
        "who'll": "who shall",
        "who'll've": "who shall have",
        "who's": "who has",
        "who've": "who have",
        "why's": "why has",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you shall",
        "you'll've": "you shall have",
        "you're": "you are",
        "you've": "you have"
        }
modify_list = [('\r\n', ' '),
               ('\n', ' '),
               ('\r', ' '),
               ('&gt;', ' '),
               ('&lt;', ' '),
               ('/__|\*|\#|(?:\[([^\]]*)\]\([^)]*\))/gm', '[link]'),
               ('https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}', '[link]'),
               ('\d+', 'NUMBER'),
               ('\[', ''),
               ('\]', ''),
               ('\/\/', ''),
               ('\.\.\.', '. ')
              ]
modify_value = {'\r\n': 1,
               '\n': 1,
               '\r': 1,
               '&gt;': 10,
               '&lt;': 10,
               '/__|\*|\#|(?:\[([^\]]*)\]\([^)]*\))/gm': 100,
               'https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}': 100,
               '\d+': 1000,
               '\[': 10000,
               '\]': 10000,
               '\/\/': 10000,
               '\.\.\.': 100000
              }
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
    for patrn in modify_list:
        new_df = df['body'].replace({patrn[0]: patrn[1]}, regex=True, inplace=False)
        modifications = list((np.where(new_df.values != df['body'].values))[0])
        df['body'] = new_df
        df['mods'][modifications] += modify_value[patrn[0]]
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
    for sentence in pbar(df.body):
        scores.append(sentence_score(basic_tokenizer(sentence)))
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
