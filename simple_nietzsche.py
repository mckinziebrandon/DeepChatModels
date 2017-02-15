from util.datasets import *
from keras.models import Sequential
from keras.layers import Embedding, Dense
import numpy as np
from pprint import pprint
import nltk
from models.minimal_rnn import MyRNN
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

# =========================================================================
# Obtain data/pre-process as needed.
# =========================================================================

token_choice = 'CHAR'

text            = get_text(DATASET).lower()
text_tokenized = nltk.word_tokenize(text) if token_choice == 'WORD' else list(text)
unique_words = sorted(list(set(text_tokenized)))

vocab_size = min(vocab_size, len(unique_words[:-REMOVE_RARE_N]))
print("VOCAB SIZE FINAL IS ", vocab_size)
print("There are {} unique words in the {} dataset.".format(len(unique_words), DATASET))
print("There are {} words total in the document.".format(len(text_tokenized)))

vocab_freqs = nltk.FreqDist(text_tokenized).most_common(vocab_size - 1)
print(vocab_freqs[:10])
print("Most common word is {} and occurs {} times".format(*vocab_freqs[0]))
print("Null rate:", float(vocab_freqs[0][1])/float(len(text_tokenized)))
vocab_words = [w for w, _ in vocab_freqs]
vocab_words.append(unknown_token)
# Replace all non-vocab words with the 'unknown' token.
text_tokenized = [w if w in vocab_words else unknown_token for w in text_tokenized]
print("Now There are {} unique words in the {} dataset.".format(len(sorted(list(set(text_tokenized)))), DATASET))
assert(len(vocab_words) == vocab_size)

print("Least common word was {} and appeared {} times.".format(*vocab_freqs[-1]))
word_to_idx = {w: i for i, w in enumerate(vocab_words)}
idx_to_word = {i: w for i, w in enumerate(vocab_words)}

# Get full text document in format of indices into vocabulary.
text_as_idx = [word_to_idx[w] for w in text_tokenized]
print("len(text_as_idx) =", len(text_as_idx))

# =========================================================================
# Make the training data.
# =========================================================================

# I think X_train should have shape (len(text), seq_len)
# Easiest to get X_train as shape (seq_len, len(text)) first, then just swap the axes.
X_train = np.array([
    [text_as_idx[i + t] for t in range(SEQ_LEN)]
    for i in np.arange(len(text_tokenized) - SEQ_LEN - 1, step=N_STEPS)
])
#X_train = np.array([[id_ch for id_ch in text_as_idx[t: len(text_tokenized) - SEQ_LEN + t]] for t in range(SEQ_LEN)])
#X_train = X_train.reshape(X_train.shape[::-1])
print("X_train is a list containing {} numpy arrays, each of shape {}.".format(len(X_train), X_train[0].shape))

y_train = np.zeros(shape=(X_train.shape[0], vocab_size))
for i in np.arange(X_train.shape[0]):
    y_train[i, text_as_idx[N_STEPS * (i + 1)]] = 1

print("y_train is a numpy array of shape {}.".format(y_train.shape))

# =========================================================================
# Prepare the model.
# =========================================================================

myRNN = MyRNN(vocab_size, seq_len=SEQ_LEN, embed_dim=256)
myRNN.summary()

# =========================================================================
# Train.
# =========================================================================

myRNN.model.load_weights('models/words_instead_V{}.h5'.format(V))
#myRNN.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=2)


# =========================================================================
# Clean up.
# =========================================================================
myRNN.model.save_weights('models/words_instead_V{}.h5'.format(V))

my_words = ("sometimes i wonder if i am the only person that exists. "
            "life is nothing but a thought. "
            "do you understand what i mean? "
            "perhaps not.")
my_words = nltk.word_tokenize(my_words)[:SEQ_LEN]
my_words = [w if w in vocab_words else unknown_token for w in my_words]

def _sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    res = a.flatten()
    res = np.log(res) / temperature
    res = np.exp(res) / (np.sum(np.exp(res)) + 1e-9)
    return np.random.multinomial(1, res, size=1).argmax()

def generate_n_words(n, temperature=0.5):
    #seq = np.array([word_to_idx[w] for w in  my_words]).reshape(1, SEQ_LEN)
    seq = np.array(text_as_idx[:SEQ_LEN]).reshape(1, SEQ_LEN)
    thoughts = [idx_to_word[i] for i in seq.flatten()]
    for i in range(n):
        print("")
        rly = 0
        pred = myRNN.model.predict(seq, batch_size=1)
        try:
            sample_ind = _sample(pred, temperature)
            #sample_ind = np.random.multinomial(1, pred.flatten(),  size=1).argmax()
        except:
            sample_ind = word_to_idx[unknown_token]
        while sample_ind == word_to_idx[unknown_token]:
            try:
                if rly > 5: sample_ind = np.random.multinomial(1, pred.flatten(),  size=1).argmax()
                else: sample_ind = _sample(pred, temperature)
            except ValueError:
                rly += 1
                if rly < 5:
                    pred /= (pred.sum() + 1e-9)
                else:
                    pred /= (pred.sum() + 1e-6)
            #sample_ind = _sample(pred, temperature)

        thoughts.append(idx_to_word[sample_ind])
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
