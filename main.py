import tensorflow as tf
import chatbot
from utils import Config

# ==============================================================================================================================
# Parser for command-line arguments.
# - Each flag below is formatted in three columns: [name] [default value] [description]
# - Each flag's value can later be accessed via: FLAGS.name
# - The flags are shown in alphabetical order (by name).
# ==============================================================================================================================

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_string("data_dir", "data", "Directory containing the data files.")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 512, "Size of each model layer.")  # TODO: just encoder? decoder too?
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.98, "TODO: DEFINE ME")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
flags.DEFINE_integer("max_train_samples", 0, "Limit training data size (0: no limit).")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")  # TODO: figure out exactly what this means.
flags.DEFINE_integer("steps_per_ckpt", 100, "How many training steps to do per checkpoint.")
flags.DEFINE_string("ckpt_dir", "out", "Directory in which checkpoint files will be saved.")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    config = Config(FLAGS)
    buckets = [(5, 10), (20, 30), (40, 50)]
    chatbot = chatbot.Chatbot(config, buckets)
