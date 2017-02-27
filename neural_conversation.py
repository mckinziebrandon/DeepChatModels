"""Train seq2seq attention model on ubuntu dialogue corpus and chat with it after."""
import os
import sys
import time
import tensorflow as tf
import data_utils
from datasets import *
from seq2seq_model import Seq2SeqModel
import logging

# ------------ File parameter choices: --------------
VOCAB_SIZE = 20000
CWD = os.getcwd()
HOME ='/home/brandon/'
DATA_DIR = HOME + 'terabyte/Datasets/ubuntu_dialogue_corpus'
CKPT_DIR = os.path.join(CWD, 'logs')
MAX_STEPS = int(1e5)
STEPS_PER_CKPT = 100
MAX_TRAIN_SAMPLES = int(1e6)
_buckets = [(5, 10), (15, 20), (25, 40)]
NUM_LAYERS = 3
SIZE = 512
# ---------------------------------------------------

def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    print("Calling model constructor . . .")
    model = Seq2SeqModel(VOCAB_SIZE, VOCAB_SIZE,
                         _buckets,
                         size=SIZE,
                         num_layers=NUM_LAYERS,
                         max_gradient_norm=5.0,
                         batch_size=64,
                         learning_rate=0.5,
                         lr_decay=0.98,
                         forward_only=forward_only,
                         dtype=tf.float32)

    print("Checking for checkpoints . . .")
    checkpoint_state  = tf.train.get_checkpoint_state(CKPT_DIR)
    if checkpoint_state and tf.train.checkpoint_exists(checkpoint_state.model_checkpoint_path):
        print("Reading model parameters from %s" % checkpoint_state.model_checkpoint_path)
        model.saver.restore(session, checkpoint_state.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def train():
    """ Train chatbot using 1-on-1 ubuntu dialogue corpus. """

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (NUM_LAYERS, SIZE))
        model = create_model(sess, False)


        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % MAX_TRAIN_SAMPLES)
        train_set, dev_set = data_utils.read_data("ubuntu",
                                                  DATA_DIR,
                                                  _buckets,
                                                  VOCAB_SIZE,
                                                  max_train_data_size=MAX_TRAIN_SAMPLES)

        # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        # The total number training samples, excluding the ones too long for our bucket choices.
        train_total_size   = float(sum(train_bucket_sizes))

        # TODO: dont 4get this is here.
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        # Translation: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        previous_losses = []
        for i_step in range(MAX_STEPS):
            # Sample a random bucket index according to the data distribution,
            # then get a batch of data from that bucket by calling model.get_batch.
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            # Get a batch and make a step.
            start_time = time.time()
            # Recall that target_weights are NOT parameter weights; they are weights in the sense of "weighted average."
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

            _, avg_perplexity, _ = model.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         target_weights,
                                         bucket_id,
                                         False)

            step_time += (time.time() - start_time) / STEPS_PER_CKPT
            loss      += avg_perplexity / STEPS_PER_CKPT

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % STEPS_PER_CKPT == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
                print("\nglobal step:", model.global_step.eval(), end="  ")
                print("learning rate: %.4f" %  model.learning_rate.eval(), end="  ")
                print("step time: %.2f" % step_time, end="  ")
                print("perplexity: %.2f" % perplexity)

                # Decrease learning rate more aggressively.
                if len(previous_losses) > 3 and loss > min(previous_losses[-3:]):
                    sess.run(model.lr_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(CKPT_DIR, "converse.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                    target_weights, bucket_id, True)
                    eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

        checkpoint_path = os.path.join(CKPT_DIR, "converse.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def decode():
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_model(sess, forward_only=True)
        model.batch_size = 1  # We decode one sentence at a time.

        # Load vocabularies.
        en_vocab_path = os.path.join(DATA_DIR, "vocab%d.from" % VOCAB_SIZE)
        fr_vocab_path = os.path.join(DATA_DIR, "vocab%d.to" % VOCAB_SIZE)

        # initialize_vocabulary returns word_to_idx, idx_to_word.
        en_vocab, _ = data_utils.initialize_vocabulary(en_vocab_path)
        _, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

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
            bucket_id = len(_buckets) - 1
            for i, bucket in enumerate(_buckets):
                if bucket[0] >= len(token_ids):
                    bucket_id = i
                    break
            else:
                logging.warning("Sentence longer than  truncated: %s", sentence)

            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              {bucket_id: [(token_ids, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
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

if __name__ == "__main__":
    #train()
    decode()

