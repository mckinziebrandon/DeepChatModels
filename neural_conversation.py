import numpy as np
import pandas as pd
from util.datasets import  *
import keras
from keras.utils            import np_utils
from keras.utils.np_utils   import to_categorical
from keras.models           import Sequential, Model
from keras.layers           import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers           import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core      import Flatten, Dense, Dropout, Lambda
from keras.regularizers     import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers       import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics          import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# ------------ File parameter choices: --------------
input_length = None #3
embed_dim = 64
vocab_size =  1000
max_sent_len = 20
hidden_dim = 256
# ---------------------------------------------------

data, dicts = get_ubuntu(vocab_size)
context_as_indices, utter_as_indices = data
print("uter shape", utter_as_indices.shape)
word_index, index_word = dicts

df_index_train = pd.DataFrame(np.hstack((context_as_indices[:, None], utter_as_indices[:, None])),
                              columns=['Context', 'Utterance'])
print(len(df_index_train))
print(df_index_train.head())

# ===================================================================
print("Building model . . . ")
# Need to setup in this way since variable-length inputs.
context_inputs   = Input(shape=(None,), dtype='int64', name='main_input')
X           = Embedding(input_dim=vocab_size, output_dim=embed_dim)(context_inputs)
C           = LSTM(output_dim=hidden_dim, input_dim=embed_dim)(X)

# y_train one-hotted is fed to utter_inputs.
# TODO: look at the diagram in ur notes.  We are almost there.
# Need to make it such that, if target output is 'WXYZ' and context is 'ABC', that mapping is:
# Encoder: A,B,C,<EOS> -> W
# Decoder:  W -> X
#           X -> Y
#           Y -> Z
#           Z -> <EOS>
utter_inputs = Input(shape=(None, vocab_size), name='utter_inputs')
decode      = LSTM(output_dim=vocab_size, input_dim=vocab_size, return_sequences=True)



model = Model(input=[context_inputs, utter_inputs], output=C)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# ===================================================================

print("Padding training sequences . . . ")
X_train = pad_sequences(context_as_indices, maxlen=max_sent_len)
_y_train = pad_sequences(utter_as_indices, maxlen=max_sent_len)
print("X_train.shape:", X_train.shape)

y_train = np.zeros(shape=(_y_train.shape[0], _y_train.shape[1], vocab_size))
for i_sent, sent in enumerate(_y_train):
    for i_word, word in enumerate(sent):
        y_train[i_sent, i_word, word] = 1
print("y_train.shape:", y_train.shape)
exit()

model.fit(X_train, y_train, batch_size=1, nb_epoch=1, verbose=1)
