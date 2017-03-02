import tensorflow as tf
import logging
import os
import sys
from utils import *
import numpy as np


def decode(chatbot, test_config):
    """Runs a chat session between the given chatbot and user."""

    chatbot.batch_size = 1  # We decode one sentence at a time.
    # Load vocabularies.
    from_vocab_path = os.path.join(test_config.data_dir, "vocab%d.from" % chatbot.vocab_size)
    to_vocab_path   = os.path.join(test_config.data_dir, "vocab%d.to" % chatbot.vocab_size)
    # initialize_vocabulary returns word_to_idx, idx_to_word.
    inputs_to_idx, _    = initialize_vocabulary(from_vocab_path)
    _, idx_to_outputs   = initialize_vocabulary(to_vocab_path)
    # Decode from standard input.
    print("Type \"exit\" to exit, obviously.")
    print("Write stuff after the \">\" below and I, your robot friend, will respond.")
    sys.stdout.write("> ")
    sys.stdout.flush()
    # Store sentence without the newline symbol.
    sentence = sys.stdin.readline()[:-1]
    while sentence:
        # Convert input sentence to token-ids.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), inputs_to_idx)
        # Get output sentence from the chatbot.
        outputs = decode_inputs(token_ids, idx_to_outputs, chatbot, test_config.temperature)
        # Print the chatbot's response.
        print(outputs)
        # Wait for next input.
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()[:-1]
        # Stop program if sentence == 'exit\n'.
        if sentence == 'exit':
            print("Fine, bye :(")
            break

def decode_inputs(inputs, idx_to_word, chatbot, temperature):
    # Which bucket does it belong to?
    bucket_id = _assign_to_bucket(inputs, chatbot.buckets)
    # Run model inference.
    output_logits = _step(chatbot, inputs, bucket_id)
    # Convert raw output to chat response & print.
    return _logits_to_outputs(output_logits, temperature, idx_to_word)

def _logits_to_outputs(output_logits, temperature, idx_word):
    """
    Args:
        output_logits: shape is [output_length, [vocab_size]]
    :return:
    """
    # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    outputs =  [_sample(l, temperature) for l in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    outputs = " ".join([tf.compat.as_str(idx_word[output]) for output in outputs]) + "."
    # Capitalize.
    outputs = outputs[0].upper() + outputs[1:]
    return outputs

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

def _assign_to_bucket(token_ids, buckets):
    """Find bucket large enough for token_ids, else warning."""
    bucket_id = len(buckets) - 1
    for i, bucket in enumerate(buckets):
        if bucket[0] >= len(token_ids):
            bucket_id = i
            break
    else:
        logging.warning("Sentence longer than  truncated: %s", len(token_ids))
    return bucket_id

def _step(chatbot, token_ids, bucket_id):
    # Get a 1-element batch to feed the sentence to the chatbot.
    encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, _, _, output_logits = chatbot.step(chatbot.sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
    return output_logits

