"""This shows how to run the new dynamic models (work in progress).
"""
import time
import tensorflow as tf
from chatbot import DynamicBot
from data import Cornell
from utils import data_utils

# ==============================================================================================================================
# Parser for command-line arguments.
# - Each flag below is formatted in three columns: [name] [default value] [description]
# - Each flag's value can later be accessed via: FLAGS.name
# - The flags are shown in alphabetical order (by name).
# ==============================================================================================================================

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", "out/logs", "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
# Integer flags -- First three only need custom values if you're especially worried about RAM.
flags.DEFINE_integer("max_train_samples", int(3e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("steps_per_ckpt", 50, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 20000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 256, "Size of each model layer.")
flags.DEFINE_integer("max_seq_len", 500, "Maximum number of words per sentence.")
# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.95, "Decay factor applied to learning rate.")
FLAGS = flags.FLAGS


def get_batched_data(data, batch_size, max_seq_len):
    encoder_sentences, decoder_sentences = data
    encoder_sentences, decoder_sentences = data_utils.batch_concatenate(
        encoder_sentences, decoder_sentences,
        batch_size, max_seq_len=max_seq_len
    )
    return encoder_sentences, decoder_sentences

if __name__ == "__main__":

    # =============================================================
    # Setup - data and chatbot creation.
    # =============================================================

    dataset = Cornell(FLAGS.vocab_size)
    bot = DynamicBot(dataset,
                     batch_size=FLAGS.batch_size,
                     max_seq_len=FLAGS.max_seq_len)
    # Don't forget to compile!
    bot.compile(reset=True)

    # Get encoder/decoder training data, with shape [None, batch_size, max_seq_len].
    encoder_sentences, decoder_sentences = get_batched_data(
        dataset.train_data, FLAGS.batch_size, FLAGS.max_seq_len
    )

    # =========================================================================
    # Train DynamicBot.
    # =========================================================================

    i_step = 0
    avg_loss = 0.0
    avg_step_time = 0.0
    try:
        while True:
            start_time      = time.time()
            step_loss       = bot(encoder_sentences[i_step], decoder_sentences[i_step])

            # Calculate running averages.
            avg_step_time  += (time.time() - start_time) / FLAGS.steps_per_ckpt
            avg_loss       += step_loss / FLAGS.steps_per_ckpt

            # Print updates in desired intervals (steps_per_ckpt).
            if i_step % FLAGS.steps_per_ckpt == 0:
                bot.save()
                print("Step {}: step time = {};  loss = {}".format(
                    i_step, avg_step_time, avg_loss))
                # Reset the running averages.
                avg_step_time = 0.0
                avg_loss = 0.0
            i_step += 1

    except (KeyboardInterrupt, SystemExit):
        print("Training halted. Cleaning up . . . ")
        bot.save()

