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
        from_vocab_path = os.path.join(config.data_dir, "vocab%d.from" % chatbot.vocab_size)
        to_vocab_path   = os.path.join(config.data_dir, "vocab%d.to" % chatbot.vocab_size)

        # initialize_vocabulary returns word_to_idx, idx_to_word.
        word_idx_from, _    = initialize_vocabulary(from_vocab_path)
        _, idx_word_to      = initialize_vocabulary(to_vocab_path)

        # Decode from standard input.
        print("Type \"exit\" to exit, obviously.")
        print("Write stuff after the \">\" below and I, your robot friend, will respond.")
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            sentence = sentence[:-1]
            # Get token-ids for the input sentence.
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), word_idx_from)
            # Which bucket does it belong to?
            bucket_id = len(chatbot.buckets) - 1
            for i, bucket in enumerate(chatbot.buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence longer than  truncated: %s", sentence)

            # ==============================================================
            # Run model inference.
            # ==============================================================

            # Get a 1-element batch to feed the sentence to the chatbot.
            encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, _, output_logits = chatbot.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

            # ==============================================================
            # Convert raw output to chat response & print.
            # ==============================================================

            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = _logits_to_indices(output_logits, config.temperature)
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]

            outputs = " ".join([tf.compat.as_str(idx_word_to[output]) for output in outputs]) + "."
            # Capitalize.
            outputs = outputs[0].upper() + outputs[1:]
            # Print out sentence corresponding to outputs.
            print(outputs)

            # ==============================================================
            # Wait for next input.
            # ==============================================================

            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            # Stop program if sentence == 'exit\n'.
            if sentence[:-1] == 'exit':
                print("Fine, bye :(")
                break

def _logits_to_indices(output_logits, temperature=0.5):
    """
    Args:
        output_logits: shape is [output_length, [vocab_size]]
    :return:
    """
    # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    return [_sample(l, temperature) for l in output_logits]

def _sample(logits, temperature):
    if temperature == 0.0:
        return int(np.argmax(logits, axis=1))
    logits = logits.flatten()
    logits = logits / temperature
    logits = np.exp(logits - np.max(logits))
    logits = logits / np.sum(logits)
    sampleID = np.argmax(np.random.multinomial(1, logits, 1))
    while sampleID == UNK_ID:
        sampleID = np.argmax(np.random.multinomial(1, logits, 1))
    return int(sampleID)

