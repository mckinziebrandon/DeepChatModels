"""Run trial run on DynamicBot with the TestData Dataset."""
import os
import numpy as np
import logging
import sys
sys.path.append("..")
from data import Cornell
from chatbot import DynamicBot
from utils import batch_concatenate


if __name__ == '__main__':

    batch_size = 32
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('TestDynamicLogger')

    # Get dataset and its properties.
    dataset = Cornell()
    encoder_sentences, decoder_sentences = dataset.train_data
    encoder_sentences = batch_concatenate(encoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)
    decoder_sentences = batch_concatenate(decoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)

    # Create the bot.
    bot = DynamicBot(dataset, batch_size=batch_size)

    num_batches = dataset.train_size // batch_size
    log.info("Dataset has %d training samples." % dataset.train_size)
    log.info("For batch size of %d, that means %d batches per epoch" % (batch_size, num_batches))
    for batch in range(min(num_batches, 5)):
        enc_inp = encoder_sentences[batch]
        dec_inp = decoder_sentences[batch]
        loss = bot(enc_inp, dec_inp)
        log.info("Batch %d: \tLoss %f" % (batch, loss))

    bot.save()

    encoder_sentences, decoder_sentences = dataset.valid_data

    # TODO: make max_seq_len take validation data into account. . . .
    encoder_sentences = encoder_sentences[:batch_size]
    decoder_sentences = decoder_sentences[:batch_size]
    assert(len(encoder_sentences) == len(decoder_sentences))

    encoder_sentences = batch_concatenate(encoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)
    decoder_sentences = batch_concatenate(decoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)

    assert(encoder_sentences.shape == decoder_sentences.shape)


    rand_batch = np.random.randint(len(encoder_sentences))

    eval_loss, outputs = bot(encoder_sentences[rand_batch],
                            decoder_sentences[rand_batch])
    print("validation loss:", eval_loss)








