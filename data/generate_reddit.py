import pandas as pd
import numpy as np
import os
import json
import re
from progressbar import ProgressBar
from multiprocessing import Process, Queue
import threading
import enchant
import pickle
from reducer import Reducer
import time

DATA_ROOT = '/Users/ivan/Documents/sp_17/reddit_data'
DATA_YEARS = ['2007']

RAW_DATA_FILES = [os.listdir(os.path.join(DATA_ROOT, 'raw_data', year)) for year in DATA_YEARS]

RAW_DATA_ABS_FILES = []

for i in range(len(DATA_YEARS)):
    for j in range(len(RAW_DATA_FILES[i])):
        if RAW_DATA_FILES[i][j].startswith('.'):
            pass
        else:
            RAW_DATA_ABS_FILES.append( os.path.join(DATA_ROOT, 'raw_data' , DATA_YEARS[i], RAW_DATA_FILES[i][j]))
RAW_DATA_FILES = RAW_DATA_ABS_FILES
RAW_DATA_ABS_FILES = []

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE   = re.compile(r"\d")


rdr = Reducer()

now = time.time()
df = rdr.load_data(RAW_DATA_FILES[0])
print("Time to read in: %d s", time.time() - now)

df = rdr.remove_faulty_row(df)
rdr.remove_extraneous_columns(df, ['archived', 'author', 'author_flair_css_class', 'author_flair_text',
        'controversiality', 'created_utc', 'distinguished', 'downs',
       'edited', 'gilded', 'id', 'retrieved_on', 'score', 'score_hidden',
       'subreddit', 'subreddit_id', 'ups'])
now = time.time()

def is_root(row):
    return row.parent_id == row.link_id

rdr.add_atribute(df, 'root', is_root)
print("Time to add root: %d s", time.time() - now)

now = time.time()

rdr.save_data(df, '/Users/ivan/Desktop/reduced_json_file.json')
print("Time to save: %d s", time.time() - now)
