import tensorflow as tf
from utils.wmt import WMT
from utils.ubuntu import Ubuntu

class Config(object):
    """Small wrapper class for the many user-specified params for training from tf FLAGS."""
    def __init__(self, FLAGS):
        self.data_dir = FLAGS.data_dir
        self.data_name = FLAGS.data_name

        if self.data_name == "ubuntu":
            self.dataset = Ubuntu(FLAGS.vocab_size)
        else:
            self.dataset = WMT(FLAGS.vocab_size)

        self.ckpt_dir = FLAGS.ckpt_dir
        self.steps_per_ckpt = FLAGS.steps_per_ckpt
        self.max_train_samples = FLAGS.max_train_samples
        #self.max_steps = FLAGS.max_steps
        self.chunk_size = FLAGS.chunk_size
        self.reset_model = FLAGS.reset_model
