"""Container classes for command-line-provided information from user."""


class TrainConfig(object):
    def __init__(self, FLAGS):
        # Should we train from scratch or load a previous model state?
        self.reset_model = FLAGS.reset_model
        # Maximum number of samples used in a training session.
        self.nb_max_samples = FLAGS.max_train_samples
        # Number of samples used per step.
        self.batch_size = FLAGS.batch_size
        # Tensorboard's logdir.
        self.log_dir    = FLAGS.log_dir
        # Directory where model's saver will place TF checkpoints.
        self.ckpt_dir = FLAGS.ckpt_dir
        # Determines how frequently checkpoints are saved.
        self.steps_per_ckpt = FLAGS.steps_per_ckpt
        # TODO: change this later when applicable.
        self.nb_epoch   = 1


class TestConfig(object):
    def __init__(self, FLAGS):
        # TODO: too chatbot-specific. Move elsewhere.
        self.temperature = FLAGS.temperature
        # For loading model parameters.
        self.ckpt_dir = FLAGS.ckpt_dir
        # For loading model graph. Question: Right?
        self.log_dir = FLAGS.log_dir
        # It makes no sense to test a newly created model.
        self.reset_model = False


