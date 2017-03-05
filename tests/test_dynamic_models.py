"""Complete mock model creation, training, and decoding from start to finish."""

import numpy as np
import logging
import sys
sys.path.append("..")

import tensorflow as tf
from utils.test_data import TestData
from chatbot.dynamic_models import DynamicBot
from utils.data_utils import batch_concatenate


if __name__ == '__main__':

    batch_size = 2

    # Get dataset and its properties.
    dataset = TestData()
    encoder_sentences, decoder_sentences = np.split(np.array(dataset.train_data), 2, axis=1)

    # TODO: the following 2 lines should not be necessary.
    encoder_sentences = list(encoder_sentences[:, 0])
    encoder_sentences = batch_concatenate(encoder_sentences, batch_size, max_seq_len=10)

    decoder_sentences = list(decoder_sentences[:, 0])
    decoder_sentences = batch_concatenate(decoder_sentences, batch_size, max_seq_len=10)

    num_batches, _, max_enc_seq = encoder_sentences.shape
    _, _, max_dec_seq = decoder_sentences.shape
    max_seq_len = max(max_enc_seq, max_dec_seq)

    print("num_batches:", num_batches)
    print("encoder_sentences max_enc_seq:", max_seq_len)
    print("dataset max_enc_seq:", dataset.max_seq_len)

    bot = DynamicBot(dataset,
                     max_seq_len=10,
                     batch_size=batch_size)

    for batch in range(num_batches):

        enc_inp = encoder_sentences[batch]
        dec_inp = decoder_sentences[batch]
        loss = bot(enc_inp, dec_inp)

        print(loss)


