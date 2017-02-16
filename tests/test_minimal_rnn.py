import unittest
from models.minimal_rnn import MyRNN
from util.datasets import *


class TestMinimalRNN(unittest.TestCase):

    def test_init(self):

        X_train, y_train = get_train_data('nietzsche', vocab_size)
        testRNN = MyRNN(1000)
        testRNN.summary()
        testRNN.fit(X_train, y_train, nb_epoch=1, verbose=1)

