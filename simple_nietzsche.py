from util.datasets import *
from keras.models import Sequential
from keras.layers import Embedding, Dense
import numpy as np
from models.minimal_rnn import MyRNN

DATASET = 'nietzsche'
SEQ_LEN = 3

# =========================================================================
# Obtain data/pre-process as needed.
# =========================================================================

text            = get_text(DATASET)
unique_chars    = sorted(list(set(text)))
vocab_size      = len(unique_chars)
print("There are {} unique characters in the {} dataset.".format(vocab_size, DATASET))
print("There are {} characters total in the document.".format(len(text)))

char_to_idx = {c: i for i, c in enumerate(unique_chars)}
idx_to_char = {i: c for i, c in enumerate(unique_chars)}

# Get full text document in format of indices into vocabulary.
text_as_idx = [char_to_idx[c] for c in text]


# 3-char model.
# I think X_train should have shape (len(text), seq_len)
# So X_train[:, 0] should give the text at 'time' 0.
# I'm gonna do this my own way: each training sample is vector of SEQ_LEN length, consisting of
# SEQ_LEN ***consecutive*** characters [ids] from the text, with the label being the next char.

# Easiest to get X_train as shape (seq_len, len(text)) first, then just swap the axes.
# Random numbers are sampled from Unif[0, VOCAB_SIZE).
#input_array = np.random.randint(VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN))
X_train = np.array([[id_ch for id_ch in text_as_idx[t: len(text) - SEQ_LEN + t]] for t in range(SEQ_LEN)])
X_train = X_train.reshape(X_train.shape[::-1])


y_train = np.zeros(shape=(len(text) - SEQ_LEN, vocab_size))
for i, id_ch in enumerate(text_as_idx[SEQ_LEN:]):
    y_train[i, id_ch] = 1

#y_train = np.array([id_ch for id_ch in text_as_idx[SEQ_LEN:]])

print("X_train is a list containing {} numpy arrays, each of shape {}.".format(len(X_train), X_train[0].shape))
print("y_train is a numpy array of shape {}.".format(y_train.shape))


# generate dummy training data
#x_train = np.random.random((1000, timesteps))
#y_train = np.random.random((1000, nb_classes))


# =========================================================================
# Prepare the model.
# =========================================================================


#model = Sequential()
# input_dim and output_dim must come first, in this order (they don't have default values).
#model.add(Embedding(input_dim=vocab_size, output_dim=42, input_length=SEQ_LEN))
#model.add(Dense(vocab_size, activation='softmax'))
#model.compile('rmsprop', 'mse')
#model.summary()

#embed_output_array = model.fit(X_train)

myRNN = MyRNN(vocab_size)
myRNN.summary()
#print(myRNN.model.input_shape)
#print(X_train.shape)


# =========================================================================
# Train.
# =========================================================================

myRNN.fit(X_train, y_train, batch_size=32, nb_epoch=1)
