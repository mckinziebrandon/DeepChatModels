import unittest

from reference.stripped_seq2seq import StrippedSimpleSeq2Seq
from util.datasets import get_train_data


class TestStrippedSeq2Seq(unittest.TestCase):

    def test_init(self):

        testRNN = StrippedSimpleSeq2Seq(input_dim=5, output_dim=8, output_length=8)
        testRNN.compile(loss='mse', optimizer='rmsprop')


        X_train, y_train = get_train_data('nietzsche')
        print(X_train.shape, y_train.shape)

        # TODO: Read through
        # https://github.com/nicolas-ivanov/debug_seq2seq/blob/master/lib/nn_model/train.py
        # to see how to format X_train and y_train.
        testRNN.fit(X_train, y_train, nb_epoch=1)
