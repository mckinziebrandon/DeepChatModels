"""Run trial run on DynamicBot with the TestData Dataset."""
import logging
import sys
sys.path.append("..")
from data import TestData
from chatbot import DynamicBot
from utils import batch_concatenate


if __name__ == '__main__':

    batch_size = 2
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('TestDynamicLogger')

    # Get dataset and its properties.
    dataset = TestData()
    encoder_sentences, decoder_sentences = dataset.train_data
    encoder_sentences = batch_concatenate(encoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)
    decoder_sentences = batch_concatenate(decoder_sentences, batch_size, max_seq_len=dataset.max_seq_len)

    print(encoder_sentences)
    print(decoder_sentences)

    # Create the bot.
    bot = DynamicBot(dataset, batch_size=batch_size)

    num_batches = dataset.train_size // batch_size
    log.info("Dataset has %d training samples." % dataset.train_size)
    log.info("For batch size of %d, that means %d batches per epoch" % (batch_size, num_batches))
    for batch in range(num_batches):
        enc_inp = encoder_sentences[batch]
        dec_inp = decoder_sentences[batch]
        loss = bot(enc_inp, dec_inp)
        log.info("Batch %d: \tLoss %f" % (batch, loss))


