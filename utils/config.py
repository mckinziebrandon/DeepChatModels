import tensorflow as tf

class Config(object):
    """Small wrapper class for the many user-specified params for training from tf FLAGS."""
    def __init__(self, FLAGS):
        self.data = FLAGS.data_dir
        self.ckpt = FLAGS.ckpt_dir
        self.steps_per_ckpt = FLAGS.steps_per_ckpt
