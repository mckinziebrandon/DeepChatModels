import unittest
from models.stripped_seq2seq import StrippedSimpleSeq2Seq
from util.datasets import get_train_data


class TestStrippedSeq2Seq(unittest.TestCase):

    def test_init(self):

        testRNN = StrippedSimpleSeq2Seq(input_dim=5, output_dim=8, output_length=8)
        testRNN.compile(loss='mse', optimizer='rmsprop')


        X_train, y_train = get_train_data('nietzsche')
        print(X_train.shape, y_train.shape)

        testRNN.fit(X_train, y_train, nb_epoch=1)
