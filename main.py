import tensorflow as tf
import chatbot
from utils import Config

# ==============================================================================================================================
# Parser for command-line arguments.
# - Each flag below is formatted in three columns: [name] [default value] [description]
# - Each flag's value can later be accessed via: FLAGS.name
# - The flags are shown in alphabetical order (by name).
# ==============================================================================================================================

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
#TEMP="/home/brandon/terabyte/Datasets/wmt"

flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("data_name", "ubuntu", "For now, either 'ubuntu' or 'wmt'.")
flags.DEFINE_string("data_dir", TEMP, "Directory containing the data files.")
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
# Integer flags -- First three only need custom values if you're especially worried about RAM.
flags.DEFINE_integer("max_train_samples", int(22e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("chunk_size", int(2e6), "")
flags.DEFINE_integer("steps_per_ckpt", 1000, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 1024, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
# Float flags -- training hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
    # Note: I'm only specifying the flags that I tend to change; more options are available!
    chatbot = chatbot.Chatbot(buckets,
                              layer_size=FLAGS.layer_size,
                              num_layers=FLAGS.num_layers)

    # The Config object stores train/test-time pertinent info, like directories.
    config = Config(FLAGS)
    if FLAGS.decode:
        chatbot.decode(config)
    else:
        chatbot.train(config)

