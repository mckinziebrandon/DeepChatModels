
import unittest
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Activation
from models.minimal_rnn import MyRNN
from utils.datasets import *
import os
import numpy as np
import seq2seq
from seq2seq.models import SimpleSeq2Seq

class TestTensorBoard(unittest.TestCase):

    def test_tb_rnn(self):
        # Function signature:
        # TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        tb_callback = TensorBoard(histogram_freq=1, write_graph=True)
        vocab_size = 1000
        X_train, y_train = get_train_data('nietzsche', vocab_size)
        testRNN = MyRNN(vocab_size)
        testRNN.fit(X_train, y_train, nb_epoch=1, verbose=1, callbacks=[tb_callback])
        os.system('tensorboard --logdir=./out')



    def test_tb_simple(self):
        # Function signature:
        # TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        tb_callback = TensorBoard(histogram_freq=1, write_graph=True)

        vocab_size = 1000
        X_train = np.random.randint(low=0, high=10, size=(500, 784))
        y_train = np.random.randint(low=0, high=10, size=(500, 10))

        #testRNN = MyRNN(vocab_size)
        testRNN = Sequential([
            Dense(32, input_dim=784, activation='relu'),
            Dense(10, activation='softmax')
        ])
        testRNN.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        testRNN.fit(X_train, y_train, nb_epoch=1, verbose=1, callbacks=[tb_callback])


        # Launch tensorboard.
        os.system('tensorboard --logdir=./out')

class PLEASETestTensorBoardPLEASE(unittest.TestCase):

    def test_tb_simple_seq2seq(self):
        # Function signature:
        # TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
        tb_callback = TensorBoard(histogram_freq=1, write_graph=True)

        vocab_size      = 1000
        input_dim       = 784
        hidden_dim      = 50
        output_length   = 10
        output_dim      = 26
        input_length = 3
        nb_samples = 500

        X_train = np.random.randint(low=0, high=10, size=(nb_samples, input_length, input_dim))
        y_train = np.random.randint(low=0, high=10, size=(nb_samples, output_length, output_dim))

        model = SimpleSeq2Seq(input_dim=input_dim, input_length=input_length, hidden_dim=hidden_dim, output_length=output_length, output_dim=output_dim)
        model.compile(loss='mse', optimizer='rmsprop')
        model.fit(X_train, y_train, nb_epoch=1, verbose=1, callbacks=[tb_callback])
        # Launch tensorboard.
        os.system('tensorboard --logdir=./out')

        # well, that wasn't very illuminating . . . . :/




