"""Used by legacy_models for decoding. Not needed by DynamicBot."""

import tensorflow as tf
import logging
import os
import sys

import utils.io_utils as data_utils
from utils.io_utils import sentence_to_token_ids, get_vocab_dicts
import numpy as np


def decode(bot, test_config):
    """Runs a chat session between the given chatbot and user."""

    # We decode one sentence at a time.
    bot.batch_size = 1
    # Load vocabularies.
    from_vocab_path = os.path.join(test_config.data_dir, "vocab%d.from" % bot.vocab_size)
    to_vocab_path   = os.path.join(test_config.data_dir, "vocab%d.to" % bot.vocab_size)
    # get_vocab_dicts returns word_to_idx, idx_to_word.
    inputs_to_idx, _    = get_vocab_dicts(from_vocab_path)
    _, idx_to_outputs   = get_vocab_dicts(to_vocab_path)
    # Decode from standard input.
    print("Type \"exit\" to exit.")
    print("Write stuff after the \">\" below and I, your robot friend, will respond.")
    sentence = _get_sentence()
    while sentence:
        # Convert input sentence to token-ids.
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(sentence), inputs_to_idx)
        # Get output sentence from the chatbot.
        outputs = decode_inputs(token_ids, idx_to_outputs, bot, test_config.temperature)
        # Print the chatbot's response.
        print(outputs)
        if test_config.teacher_mode:
            print("What should I have said?")
            feedback = _get_sentence()
            feedback_ids = sentence_to_token_ids(tf.compat.as_bytes(feedback), inputs_to_idx)
            outputs = train_on_feedback(bot, token_ids, feedback_ids, idx_to_outputs, test_config.temperature)
            print("Okay. Let me try again:\n", outputs)
        # Wait for next input.
        sentence = _get_sentence()
        # Stop program if sentence == 'exit\n'.
        if sentence == 'exit':
            print("Fine, bye :(")
            break


def decode_inputs(inputs, idx_to_word, chatbot, temperature):
    # Which bucket does it belong to?
    bucket_id = _assign_to_bucket(inputs, chatbot.buckets)
    # Get a 1-element batch to feed the sentence to the chatbot.
    data = {bucket_id: [(inputs, [])]}
    encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch(data, bucket_id)
    # Get output logits for the sentence.
    _, _, _, output_logits = chatbot.step(encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
    # Convert raw output to chat response & print.
    return _logits_to_outputs(output_logits, temperature, idx_to_word)


def train_on_feedback(chatbot, input_ids, feedback_ids, idx_to_outputs, temperature):
    bucket_id = _assign_to_bucket(feedback_ids, chatbot.buckets)
    data = {bucket_id: [(input_ids, feedback_ids)]}
    enc_in, dec_in, weights = chatbot.get_batch(data, bucket_id)
    # Jack up learning rate & make sure robot learned its lesson.
    chatbot.sess.run(chatbot.learning_rate.assign(0.7))
    for _ in range(10):
        # LEARN YOU FOOL, LEARN. :)
        chatbot.step(enc_in, dec_in, weights, bucket_id, False)
    return decode_inputs(input_ids, idx_to_outputs, chatbot, temperature)


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
    if temperature < 0.5:
        return int(np.argmax(logits, axis=1))
    logits = logits.flatten()
    logits = logits / temperature
    logits = np.exp(logits - np.max(logits))
    logits = logits / np.sum(logits)
    sampleID = np.argmax(np.random.multinomial(1, logits, 1))
    while sampleID == data_utils.UNK_ID:
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


def _get_sentence():
    sys.stdout.write("> ")
    sys.stdout.flush()
    return sys.stdin.readline()[:-1]
