"""Run trial run on DynamicBot with the TestData Dataset."""
import os
import numpy as np
import logging
import sys
sys.path.append("..")
from data import Cornell
from chatbot import DynamicBot
from utils import batch_concatenate

def get_batched_data(data, batch_size, max_seq_len):
    encoder_sentences, decoder_sentences = data
    encoder_sentences, decoder_sentences = batch_concatenate(
        encoder_sentences, decoder_sentences,
        batch_size, max_seq_len=max_seq_len
    )
    return encoder_sentences, decoder_sentences

if __name__ == '__main__':

    batch_size = 32
    max_seq_len = 500
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('TestDynamicLogger')

    # =========================================================================
    # Get training data and reformat as desired.
    # =========================================================================

    log.info("Retrieving the Cornell dataset...")
    dataset = Cornell()
    encoder_sentences, decoder_sentences = get_batched_data(
        dataset.train_data, batch_size, max_seq_len
    )
    log.info("...Dataset retrieved.")

    # =========================================================================
    # Set up DynamicBot.
    # =========================================================================

    # Create the bot.
    log.info("Creating DynamicBot...")
    bot = DynamicBot(dataset, batch_size=batch_size, max_seq_len=max_seq_len)
    log.info("...DynamicBot created")
    log.info("Compiling DynamicBot...")
    bot.compile()
    log.info("...DynamicBot compiiled.")

    # =========================================================================
    # Train DynamicBot.
    # =========================================================================

    num_batches = dataset.train_size // batch_size
    log.info("Dataset has %d training samples." % dataset.train_size)
    log.info("For batch size of %d, that means %d batches per epoch" % (batch_size, num_batches))
    for batch in range(min(num_batches, 5)):
        enc_inp = encoder_sentences[batch]
        dec_inp = decoder_sentences[batch]
        loss = bot(enc_inp, dec_inp)
        log.info("Batch %d: \tLoss %f" % (batch, loss))

    bot.save()

    # =========================================================================
    # Get validation data and reformat as desired.
    # =========================================================================

    encoder_sentences, decoder_sentences = get_batched_data(
        dataset.valid_data, batch_size, max_seq_len
    )
    eval_loss, outputs = bot(encoder_sentences[0], decoder_sentences[0], forward_only=True)
    print("validation loss:", eval_loss)








