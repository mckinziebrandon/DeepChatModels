from utils.datasets import *
from datetime import datetime
import sys
import numpy as np
from keras.preprocessing.text import *
import csv as csv
import itertools

import nltk

vocabulary_size = 4000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

V=6
vocab_size = 4000
DATASET = 'nietzsche'
SEQ_LEN = 15
N_STEPS=6
REMOVE_RARE_N=15
token_choice = 'WORD'
# =========================================================================
# Obtain data/pre-process as needed.
# =========================================================================

# Get the word list.
sentences = nltk.sent_tokenize(get_text(DATASET))

# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(list(word_freq.items())))

# Get the most common words and build index_word and word_index vectors
vocab = word_freq.most_common(vocabulary_size - 1)
index_word = [x[0] for x in vocab]
index_word.append(unknown_token)
word_index = dict([(w, i) for i, w in enumerate(index_word)])

print("Using vocabulary size %d." % vocabulary_size)
print("Least frequent word in vocab: '%s', which appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_index else unknown_token for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = np.asarray([[word_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_index[w] for w in sent[1:]] for sent in tokenized_sentences])
print(type(tokenized_sentences))


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_index[sentence_end_token]:
        next_word_probs, _ = model.forward_pass(new_sentence)
        sampled_word = word_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_word[x] for x in new_sentence[1:-1]]
    return sentence_str

def generate(model, n_sentences=1, min_length=5):
    for i in range(n_sentences):
        i_try = 0
        sent = []
        while len(sent) < min_length or i_try > 5:
            i_try += 1
            sent = generate_sentence(model)
        print(" ".join(sent))
        print("")



from models_from_scratch.rnn import RNN
np.random.seed(10)
rnn = RNN(vocab_size=vocabulary_size,
                hidden_size=10,
                bptt_truncate=256,
                dicts=[word_index, index_word])

preds = rnn.predict(X_train[10])

print( "Expected Loss for random predictions: %f" % np.log(vocabulary_size))
print( "Actual loss: %f" % rnn.loss(X_train[:100], y_train[:100]))

ind = np.random.randint(0, X_train.shape[0], size=1000)
losses = train_with_sgd(rnn, X_train[ind], y_train[ind], nepoch=10, evaluate_loss_after=1)
generate(rnn, n_sentences=10, min_length=10)
