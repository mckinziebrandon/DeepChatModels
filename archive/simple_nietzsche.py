from util.datasets import *
from keras.models import Sequential
from keras.layers import Embedding, Dense
import numpy as np
from pprint import pprint
import nltk
from models.minimal_rnn import MyRNN
from keras.preprocessing.text import *
import pdb


V=6
vocab_size = 4000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
DATASET = 'nietzsche'
SEQ_LEN = 15
N_STEPS=6
REMOVE_RARE_N=15
token_choice = 'WORD'


# =========================================================================
# Obtain data/pre-process as needed.
# =========================================================================

# Get the word list.
text_as_words = text_to_word_sequence(get_text(DATASET))

# Set up the Tokenizer object.
tokenizer = Tokenizer(nb_words=vocab_size)
tokenizer.fit_on_texts(text_as_words)

# Get the dictionaries between words <-> indices.
word_index = tokenizer.word_index
index_word = {i:w for w, i in word_index.items()}

# Get full text document in format of indices into vocabulary.
text_as_idx = [word_index[w] for w in text_as_words]
print("len(text_as_idx) =", len(text_as_idx))

# =========================================================================
# Make the training data.
# =========================================================================
vocab_size = len(word_index)
print("new vocab size . . . ", vocab_size)

# I think X_train should have shape (len(text), seq_len)
# Easiest to get X_train as shape (seq_len, len(text)) first, then just swap the axes.
X_train = np.array([
    [text_as_idx[i + t] for t in range(SEQ_LEN)]
    for i in np.arange(len(text_as_words) - SEQ_LEN - 1, step=N_STEPS)
])
print("X_train is a list containing {} numpy arrays, each of shape {}.".format(len(X_train), X_train[0].shape))

y_train = np.zeros(shape=(X_train.shape[0], vocab_size))
for i in np.arange(X_train.shape[0]):
    y_train[i, text_as_idx[N_STEPS * (i + 1)]] = 1

print("y_train is a numpy array of shape {}.".format(y_train.shape))

# =========================================================================
# Prepare the model.
# =========================================================================

myRNN = MyRNN(vocab_size, seq_len=SEQ_LEN, embed_dim=128)
myRNN.summary()

# =========================================================================
# Train.
# =========================================================================

myRNN.model.load_weights('models/words_instead_V{}.h5'.format(V))
myRNN.fit(X_train, y_train, batch_size=128, nb_epoch=1, verbose=1)


# =========================================================================
# Clean up.
# =========================================================================
myRNN.model.save_weights('models/words_instead_V{}.h5'.format(V))


def _sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    res = a.flatten()
    res = np.log(res) / temperature
    res = np.exp(res) / (np.sum(np.exp(res)) + 1e-9)
    return np.random.multinomial(1, res, size=1).argmax()

def generate_n_words(n, temperature=0.5):
    #seq = np.array([word_to_idx[w] for w in  my_words]).reshape(1, SEQ_LEN)
    seq = np.array(text_as_idx[:SEQ_LEN]).reshape(1, SEQ_LEN)
    thoughts = [index_word[i] for i in seq.flatten()]
    for i in range(n):
        print("")
        rly = 0
        pred = myRNN.model.predict(seq, batch_size=1)
        sample_ind = _sample(pred, temperature)

        thoughts.append(index_word[sample_ind])
        seq = np.hstack((seq[:, 1:].flatten(), [sample_ind]))
        seq = seq.reshape(1, SEQ_LEN)
    return thoughts

n = 5 * SEQ_LEN
for t in [0.5]:
    print("\n ====================== temp = {} =======================".format(t))
    if token_choice == 'WORD':
        sentences = nltk.sent_tokenize(' '.join(generate_n_words(n, t)))
    else:
        sentences = ''.join(generate_n_words(n, t))

    print("\n")
    print(sentences)

exit()
