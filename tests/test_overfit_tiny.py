from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from pprint import pprint
import sys
import tensorflow as tf
import time
import logging
sys.path.insert(0, os.path.abspath('..'))
from utils import *
from chatbot import DynamicBot
from data import TestData
from utils import io_utils

# ==================================================================================================
# Goal: Make sure models can overfit and essentially memorize the small TestData set. This is
# a suggested debugging technique in the textbook "Deep Learning" by Goodfellow et al.
#
# ...Yay. It worked. Good robot.
# ==================================================================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('TestOverfitLogger')

    # Set simple input parameters.
    chat_session = False
    reset = True if not chat_session else False
    state_size = 1024
    embed_size = 64
    learning_rate = 0.02
    batch_size = 4
    temperature = 0.2

    # All datasets follow the same API, found in data/_dataset.py
    log.info("Setting up dataset.")
    dataset = TestData(vocab_size=64)

    dataset.convert_to_tf_records(dataset.paths['from_train'], dataset.paths['to_train'], 2)

    exit()

    # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     batch_size=batch_size,
                     ckpt_dir="out",
                     dropout_prob=0.0,
                     embed_size=embed_size,
                     learning_rate=learning_rate,
                     lr_decay=0.999,
                     state_size=state_size,
                     steps_per_ckpt=12,
                     temperature=temperature,
                     is_chatting=chat_session)

    # Don't forget to compile!
    log.info("Compiling DynamicBot.")
    bot.compile(max_gradient=10.0, reset=reset)
    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"
    print("Training bot. CTRL-C to stop training.")
    bot.train(dataset, nb_epoch=5)


def mini_train(bot, dataset):
    wi_from, iw_from = io_utils.get_vocab_dicts(dataset.paths['from_vocab'])
    wi_to, iw_to = io_utils.get_vocab_dicts(dataset.paths['to_vocab'])

    def print_as_words(e, d):
        print("Next batch:")
        for j in range(len(e)):
            context =  " ".join([tf.compat.as_str(iw_from[i]) for i in e[j][::-1]])
            response =  " ".join([tf.compat.as_str(iw_to[i]) for i in d[j]])
            print("\tencoder:", context)
            print("\tdecoder:", response)
            print()
        sys.stdin.readline()

    print("Would you like to see batch sentences as they're fed in? [y/n] ")
    wants_outputs = io_utils.get_sentence()
    i_step = 0
    while True:
        avg_loss =  0.0
        # Create data generators for feeding inputs to step().
        train_gen = dataset.train_generator(bot.batch_size)
        valid_gen = dataset.valid_generator(bot.batch_size)
        for encoder_batch, decoder_batch in train_gen:

            if wants_outputs == 'y': print_as_words(encoder_batch, decoder_batch)

            start_time = time.time()
            summaries, step_loss, _ = bot.step(encoder_batch, decoder_batch)
            avg_loss       += step_loss / bot.steps_per_ckpt
            if i_step % bot.steps_per_ckpt == 0:
                log.info("Step %d:" % i_step)
                log.info("training loss = %.3f" % avg_loss)
                try:
                    summaries, eval_loss, _ = bot.step(*next(valid_gen))
                except StopIteration:
                    log.warning("StopIteration returned by validation generator. Resetting.")
                    valid_gen = dataset.valid_generator(bot.batch_size)
                    summaries, eval_loss, _ = bot.step(*next(valid_gen))
                log.info("\tValidation loss = %.3f" % eval_loss)
                sys.stdout.flush()
                avg_loss = 0.0
            i_step += 1
