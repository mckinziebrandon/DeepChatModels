import numpy as np
import keras
from keras                  import backend as K
from keras.utils.data_utils import get_file
from keras.utils            import np_utils
from keras.utils.np_utils   import to_categorical
from keras.models           import Sequential, Model
from keras.layers           import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers           import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core      import Flatten, Dense, Dropout, Lambda


class MyRNN:

    def __init__(self, vocab_size, seq_len=3, embed_dim=42):

        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.embed_dim = embed_dim
        print('wtf')

        #[Input(shape=(1,), dtype='int64', name='input{}'.format(t)) for t in range(self.seq_len)]
        # Output shape of embedding layer: (batch_size, seq_len, embed_dim)
        self.model = Sequential([
            # First two args: input_DIM and output_DIM (not shape!)
            Embedding(self.vocab_size, self.embed_dim, input_length=self.seq_len),
            LSTM(32, return_sequences=True, input_shape=(self.seq_len, self.embed_dim)), # Returns seq_len number of vectors.
            LSTM(32), #, input_shape=(self.seq_len, self.embed_dim)), # Returns one vector (of dimension embed_dim).
            Dropout(0.5),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # Get the keras input layers.
        #self.inputs = self._create_inputs()
        # Send inputs through Keras embedding layers.
        #self.embeddings = self._create_embeddings()
        # Get the parameter weight matrices.
        #self.params = self._create_params()
        # Get the hidden states, indexed by t (self.seq_len total).
        #self.h = self._compute_hidden_states(self.embeddings)
        # Get the (single, for now) output state.
        #self.output = self._compute_output_states()
        ## Finally, define and store the keras model.
        #self.model = self._define_model(model_type)


    def fit(self, x, y, **kwargs):
        self.model.fit(x, y, **kwargs)


    def predict(self, idx_chars, **kwargs):

        x_test = np.array(idx_chars).reshape(1, self.seq_len)
        # ============================
        p = self.model.predict(x_test, **kwargs)
        # ============================
        return p


    def _define_model(self, model_type='manual', learning_rate=1e-6):
        return Model(input=self.inputs, output=self.output)

    def _create_inputs(self):
        """ Create one input layer for each timestep (out of seq_len steps total). """
        return [Input(shape=(1,), dtype='int64', name='input{}'.format(t)) for t in range(self.seq_len)]

    def _create_embeddings(self, n_latent_factors=42):
        """
        keras.layers.Embedding(...) info:
            Input shape: (nb_samples, sequence_length)
            Output shape: (nb_samples, sequence_length, output_dim)
        """
        embeddings = {}
        for t in range(self.seq_len):
            # Create the layer.
            embeddings[t] = Embedding(input_dim=self.vocab_size,
                                      output_dim=n_latent_factors,
                                      input_length=1)
            # Specify which input the layer acts upon.
            embeddings[t] = embeddings[t](self.inputs[t])
            # Flatten the output, since is fed into a Dense layer.
            embeddings[t] = Flatten()(embeddings[t])
        return embeddings

    def _create_params(self):
        # Parameters: input -> hidden.
        U = Dense(output_dim=self.n_hid, activation='relu')
        # Parameters: hidden -> hidden.
        W = Dense(output_dim=self.n_hid, activation='tanh')
        # Parameters: hidden -> output.
        V = Dense(output_dim=self.vocab_size, activation='softmax')
        return {'U': U, 'W': W, 'V': V}

    def _compute_hidden_states(self, embedded_xs):
        """
        Description:
            Use the typical RNN formula below to compute
            hidden states at each time step.
            h[t] = activation_func(Ux[t] + Wh[t - 1])
        Notes:
            merge([l1, l2, l3, ...]) returns ELEMENT-WISE sum over layers.
        """

        # Hidden state, indexed by time.
        h = [0 for _ in range(self.seq_len)]
        # Initial is computed just from first embedded input.
        h[0] = self.params['U'](embedded_xs[0])
        for t in range(1, self.seq_len):
            h[t] = merge([self.params['U'](embedded_xs[t]),
                          self.params['W'](h[t - 1])])
        return h

    def _compute_output_states(self):
        return self.params['V'](self.h[self.seq_len - 1])

    def summary(self):
        self.model.summary()

