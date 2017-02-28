import numpy as np
import pdb
import operator


class RNN(object):
    """Simple RNN implemention with numpy.
    Attributes:
        U: Connections between Inputs -> Hidden. Shape = (hidden_size, vocab_size)
        V: Connections between Hidden -> Output. Shape = (vocab_size, hidden_size)
        W: Connections between Hidden -> Hidden. Shape = (hidden_size, hidden_size)
    """

    def __init__(self, vocab_size=4, hidden_size=3, bptt_truncate=4, dicts=[], init_weights=True):
        self.n_vocab = vocab_size
        self.n_hid = hidden_size
        self.bptt_truncate = bptt_truncate
        self.char_to_ix, self.ix_to_char = dicts

        # _____________ Model Parameters. ______________
        # Index convention: Array[i, j] is from neuron j to neuron i.
        # Init values based on number of incoming connections from the *previous* layer.
        if init_weights:
            init = {'in': np.sqrt(1. / vocab_size), 'hid': np.sqrt(1. / hidden_size)}
            self.U = np.random.uniform(- init['in'], init['in'], size=(hidden_size, vocab_size))
            self.V = np.random.uniform(- init['hid'], init['hid'], size=(vocab_size, hidden_size))
            self.W = np.random.uniform(- init['hid'], init['hid'], size=(hidden_size, hidden_size))

    def _step(self, x, o, s, t):
        # Indexing U by x[t] is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(self.U[:, x[t]] + self.W @ s[t - 1])
        o[t] = self.V @ s[t]
        # Technically, this softmax(o) == \hat{y}, NOT O. Whatevs.
        o[t] = np.exp(o[t]) / np.exp(o[t]).sum()
        return o[t], s[t]

    def forward_pass(self, x, verbose=False):
        """
        Args:
            x:  a list of word indices. We keep it this way to avoid converting to a
                ridiculously large one-hot encoded matrix.
            step: function(x, s, t)
        Returns:
            o: output probabilities over all inputs in x. shape: (len(x), n_vocab)
            s: hidden states at each time step. shape: (len(x) + 1, n_hid)
        """
        n_steps = len(x)
        # Save hidden states in s because need them later. (extra element for initial hidden state)
        s = np.zeros((n_steps + 1, self.n_hid))
        s[-1] = np.zeros(self.n_hid)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((n_steps, self.n_vocab))
        # Feed in each word of x sequentially.
        for t in np.arange(n_steps):
            o[t], s[t] = self._step(x, o, s, t)
        return [o, s]

    def predict(self, x):
        """P
        Args:
            x: training sample sentence.
        Returns:
            max_out_ind: [indices of] most likely words, given the input sentence.
        """
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_pass(x)
        max_out_ind = np.argmax(o, axis=1)
        # TODO: why not just return the pred_words instead & not print . . .
        pred_words = [self.ix_to_char[i] for i in max_out_ind]
        print('Preds at each time step:\n', ' '.join(pred_words))
        return max_out_ind

    def loss(self, x, y, norm=True):
        """
        Args:
            x: list of input sentences (as indices).
            y: list of target sentences (as indices).
            norm: if True, divide total loss by number of training words.
        """

        # Shouldn't N = len(y) . . . ?
        N = np.sum((len(y_i) for y_i in y)) if norm else 1
        total_loss = 0
        for i in np.arange(len(y)):
            o, s = self.forward_pass(x[i])
            # Extract our predicted probabilities for the actual labels y.
            predicted_label_prob = o[np.arange(len(y[i])), y[i]]
            # Increment loss. Multiply by 1. to remind of interp y_n = 1 for truth else 0.
            total_loss += - 1. * np.sum(np.log(predicted_label_prob))
        return total_loss / N

    def bptt(self, x, y):
        """Backpropagation Through Time.

        Args:
            x: list. single sentence of IDs.
            y: list. single sentence of IDs.
        """
        n_words = len(y)  # in the single sentence of y.
        # Perform forward propagation
        softmax_out, s = self.forward_pass(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = softmax_out
        delta_o[np.arange(len(y)), y] -= 1.
        # Countdown backwards from T.
        for t in np.arange(n_words)[::-1]:
            dLdV += np.outer(delta_o[t], s[t])
            delta_h = self.V.T @ delta_o[t]
            # Step backwards in time for either btt_truncate steps or hit 0, whichever comes first.
            for bptt_step in np.arange(max(1, t - self.bptt_truncate), t + 1)[::-1]:
                tmp   = (1. - s[bptt_step] ** 2)
                dLdW += tmp * np.outer(delta_h, s[bptt_step - 1])
                # Next line == pure genius. I checked the math myself and it is indeed correct. Mind blown.
                dLdU[:, x[bptt_step]] += tmp * delta_h
                delta_h = self.W.T.dot(delta_h) * tmp + self.V.T @ delta_o[bptt_step]

        return [dLdU, dLdV, dLdW]

    # Performs one step of SGD.
    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    from datetime import datetime
    import sys
