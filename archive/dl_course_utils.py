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
from keras.regularizers     import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers       import SGD, RMSprop, Adam
from keras.utils.layer_utils import layer_from_config
from keras.metrics          import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

class MyRNN:

    def __init__(self,
            vocab_size,
            dicts=[],
            model_type='manual',
            seq_len=3,
            n_hid=256,
            n_fac=42):

        self.n_hid      = n_hid
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.char_indices, self.indices_char = dicts
        self.n_latent_factors = n_fac
        self.model_type = model_type

        if model_type == 'manual':
            # Get the keras input layers.
            self.inputs = self._create_inputs()
            # Send inputs through Keras embedding layers.
            self.embeddings = self._create_embeddings()
            # Get the parameter weight matrices.
            self.params = self._create_params()
            # Get the hidden states, indexed by t (self.seq_len total).
            self.h = self._compute_hidden_states(self.embeddings)
            # Get the (single, for now) output state.
            self.output = self._compute_output_states()

        # Finally, define and store the keras model.
        self.model = self._define_model(model_type)


    def fit(self, x, y, **kwargs):
        if self.model_type == 'SeqRNN':
            zeros = np.tile(np.zeros(42),(len(x[0]),1))
            print(zeros.shape)
            assert(zeros.shape==(75109, 42))
            self.model.fit([zeros] + x, y, **kwargs)
        else:
            self.model.fit(x, y, **kwargs)


    def predict(self, chars, unique_chars):
        id_chars = [self.char_indices[c] for c in chars]
        if self.model_type == 'manual':
            test_inputs = [np.array(i)[np.newaxis] for i in id_chars]
            p = self.model.predict(test_inputs)
        elif self.model_type == 'SimpleRNN':
            test_inputs = np.array(id_chars)[np.newaxis,:]
            p = self.model.predict(test_inputs)[0]
        elif self.model_type == 'SeqRNN':
            test_inputs = [np.array(i)[np.newaxis] for i in id_chars]
            p = self.model.predict([np.zeros(42)[np.newaxis,:]] + test_inputs)
            return [unique_chars[np.argmax(o)] for o in p]

        idx_highest_probability = np.argmax(p)
        return unique_chars[idx_highest_probability]

    def _define_model(self, model_type='manual', learning_rate=1e-6):
        if model_type == 'manual':
            model = Model(input=self.inputs, output=self.output)
        elif model_type == 'SimpleRNN':
            model = Sequential([
                    Embedding(self.vocab_size,
                              self.n_latent_factors,
                              input_length=self.seq_len),
                    SimpleRNN(self.n_hid, activation='relu', inner_init='identity'),
                    Dense(self.vocab_size, activation='softmax')
                ])
        elif model_type == 'SeqRNN':

            n_hidden=256
            dense_in = Dense(n_hidden, activation='relu')
            dense_hidden = Dense(n_hidden, activation='relu', init='identity')
            dense_out = Dense(self.vocab_size, activation='softmax', name='output')

            def embedding_input(name, n_in, n_out):
                inp = Input(shape=(1,), dtype='int64', name=name+'_in')
                emb = Embedding(n_in, n_out, input_length=1, name=name+'_emb')(inp)
                return inp, Flatten()(emb)

            c_ins = [embedding_input('c'+str(n),
                                     self.vocab_size,
                                     self.n_latent_factors)\
                        for n in range(self.seq_len)]

            inp1 = Input(shape=(self.n_latent_factors,), name='zeros')

            hidden = dense_in(inp1)
            outs = []
            cs = self.seq_len
            for i in range(8):
                c_dense = dense_in(c_ins[i][1])
                hidden = dense_hidden(hidden)
                hidden = merge([c_dense, hidden], mode='sum')
                outs.append(dense_out(hidden))
            model = Model([inp1] + [c[0] for c in c_ins], outs)

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam())
        #model.optimizer.lr = learning_rate
        return model

    def _create_inputs(self):
        """ Create one input layer for each timestep (out of seq_len steps total). """

        return [Input(shape=(1,), dtype='int64', name='input{}'.format(t)) \
                for t in range(self.seq_len)]

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

    @property
    def summary(self):
        self.model.summary()

class TextHelper:
    # SFold
    def __init__(self, text):
        self.text = text

    def text(self):
        return self.text

    @property
    def text_length(self):
        return len(self.text)

    @property
    def unique_chars(self):
        return sorted(list(set(self.text)))

    @property
    def num_chars(self):
        return len(self.unique_chars)
    # EFold


