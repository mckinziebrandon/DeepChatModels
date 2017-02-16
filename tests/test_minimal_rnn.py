import unittest
from models.minimal_rnn import MyRNN


class TestMinimalRNN(unittest.TestCase):

    def test_init(self):

        testRNN = MyRNN(1000)
        testRNN.summary()

