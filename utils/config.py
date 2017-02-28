import tensorflow as tf

class Config(object):
    """Small wrapper class for the many user-specified params for training from tf FLAGS."""
    def __init__(self, FLAGS):
        self.data_dir = FLAGS.data_dir
        self.ckpt_dir = FLAGS.ckpt_dir
        self.steps_per_ckpt = FLAGS.steps_per_ckpt
        self.data_name = FLAGS.data_name
        self.max_train_samples = FLAGS.max_train_samples
        #self.max_steps = FLAGS.max_steps
        self.chunk_size = FLAGS.chunk_size
        self.reset_model = FLAGS.reset_model
