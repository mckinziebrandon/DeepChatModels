import numpy as np
from keras.models           import Sequential, Model
from keras.layers           import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers.core      import Flatten, Dense, Dropout, Lambda


class MyRNN:

    def __init__(self, vocab_size, seq_len=3, embed_dim=42):

        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.embed_dim = embed_dim

        self.model = Sequential([
            # First two args: input_DIM and output_DIM (not shape!)
            Embedding(self.vocab_size, self.embed_dim, input_length=self.seq_len),
            LSTM(32, return_sequences=True, input_shape=(self.seq_len, self.embed_dim)), # Returns seq_len number of vectors.
            LSTM(32), #, input_shape=(self.seq_len, self.embed_dim)), # Returns one vector (of dimension embed_dim).
            Dropout(0.5),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)


    def predict(self, idx_chars, **kwargs):
        x_test = np.array(idx_chars).reshape(1, self.seq_len)
        return self.model.predict(x_test, **kwargs)

    def summary(self):
        self.model.summary()

