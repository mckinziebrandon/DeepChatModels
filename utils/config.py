"""Container classes for command-line-provided information from user.
Note: Only used by legacy_models.
"""



class TrainConfig(object):
    def __init__(self, FLAGS):
        self.reset_model        = FLAGS.reset_model
        self.max_train_samples  = FLAGS.max_train_samples
        self.batch_size         = FLAGS.batch_size
        self.log_dir            = FLAGS.log_dir
        self.ckpt_dir           = FLAGS.ckpt_dir
        self.steps_per_ckpt     = FLAGS.steps_per_ckpt
        self.nb_epoch           = FLAGS.nb_epoch


class TestConfig(object):
    def __init__(self, FLAGS):
        self.temperature    = FLAGS.temperature
        self.ckpt_dir       = FLAGS.ckpt_dir
        self.log_dir        = FLAGS.log_dir
        self.teacher_mode   = True
        # Protecting the users from themselves . . .
        self.reset_model    = False
