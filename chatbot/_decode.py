import tensorflow as tf
import logging
import os
import sys
from utils import *
import numpy as np


def _decode(chatbot, config):

    with chatbot.sess as sess:
        chatbot.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(config.data_dir, "vocab%d.from" % chatbot.vocab_size)
        fr_vocab_path = os.path.join(config.data_dir, "vocab%d.to" % chatbot.vocab_size)

        # initialize_vocabulary returns word_to_idx, idx_to_word.
        en_vocab, _ = initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = initialize_vocabulary(fr_vocab_path)

        # Decode from standard input.
        print("Type \"exit\" to exit, obviously.")
        print("Write stuff after the \">\" below and I, your robot friend, will translate it to robot French.")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            sentence = sentence[:-1]
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), en_vocab)
            # Which bucket does it belong to?
            bucket_id = len(chatbot.buckets) - 1
            for i, bucket in enumerate(chatbot.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence longer than  truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the chatbot.
            encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = chatbot.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
               outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            # Stop program if sentence == 'exit\n'.
            if sentence[:-1] == 'exit':
                print("Fine, bye :(")
                break
