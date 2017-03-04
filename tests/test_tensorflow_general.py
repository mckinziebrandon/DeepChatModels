import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
import numpy as np
import time
from utils.data_utils import *
from utils.config import TrainConfig

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
#TEMP="/home/brandon/terabyte/Datasets/wmt"
BASE='/home/brandon/Documents/seq2seq_projects/tests'


flags = tf.app.flags
# String flags -- directories and dataset name(s).
flags.DEFINE_string("data_name", "ubuntu", "For now, either 'ubuntu' or 'wmt'.")
flags.DEFINE_string("data_dir", TEMP, "Directory containing the data files.")
flags.DEFINE_string("ckpt_dir", BASE+'/out', "Directory in which checkpoint files will be saved.")
flags.DEFINE_string("log_dir", BASE+'/out/logs' , "Directory in which checkpoint files will be saved.")
# Boolean flags.
flags.DEFINE_boolean("reset_model", True, "wipe output directory; new params")
flags.DEFINE_boolean("decode", False, "If true, will activate chat session with user.")
# Integer flags -- First three only need custom values if you're especially worried about RAM.
flags.DEFINE_integer("max_train_samples", int(1e6), "Limit training data size (0: no limit).")
flags.DEFINE_integer("steps_per_ckpt", 50, "How many training steps to do per checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size to use during training.")
flags.DEFINE_integer("vocab_size", 40000, "English vocabulary size.")
flags.DEFINE_integer("layer_size", 128, "Size of each model layer.")
flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
# Float flags -- hyperparameters.
flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
flags.DEFINE_float("lr_decay", 0.95, "Decay factor applied to learning rate.")
flags.DEFINE_float("max_gradient", 5.0, "Clip gradients to this value.")
flags.DEFINE_float("temperature", 0.0, "Sampling temperature.")
FLAGS = flags.FLAGS

class TestTFGeneral(unittest.TestCase):
    """ --------  Notes: unittest module. --------
      Regarding this file:
      - TestTensorflowSaver is a single test case.
      - Any methods of it beginning with 'test' will be used by the test runner.
      - To take advantage of the testrunner, use self.assert{Equals, True, etc..} methods.

      More general:
      - @unittest.skip(msg) before a method will skip it.
      - right clicking inside the method and running DOES run that test only!
    """
    def test_scope(self):
        """Ensure I understand nested scoping in tensorflow."""
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger('TestTFGeneral.test_scope')
        with tf.variable_scope("scope_level_1") as scope_one:
            var_scope_one = tf.get_variable("var_scope_one")
            # Check retrieved scope name against scope_one.name.
            actual_name = tf.get_variable_scope().name
            log.info("\nscope_one.name: {}".format(scope_one.name))
            log.info("Retrieved name: {}".format(actual_name))
            self.assertEqual(scope_one.name, actual_name)
            self.assertTrue(scope_one.reuse == False)

            # Explore:
            # - How do variable names change with scope?
            # - How does reusing variables work?
            with tf.variable_scope("scope_level_2") as scope_two:
                # Check retrieved scope name against scope_two.name.
                actual_name = tf.get_variable_scope().name
                log.info("\nscope_two.name: {}".format(scope_two.name))
                log.info("Retrieved name: {}".format(actual_name))
                self.assertEqual(scope_two.name, actual_name)
                self.assertTrue(scope_two.name == scope_one.name + "/scope_level_2")
                # Example of reuse variables.
                self.assertTrue(scope_two.reuse == False)
                scope_two.reuse_variables()
                self.assertTrue(scope_two.reuse == True)
                # Reuse variables behavior is inherited:
                with tf.variable_scope("scope_level_3") as scope_three:
                    self.assertTrue(scope_three.reuse == True)
                    self.assertIs(tf.get_variable_scope(), scope_three)

        # Example: opening a variable scope with a previous scope, instead of explicit string.
        with tf.variable_scope(scope_one, reuse=True):
            x = tf.get_variable("var_scope_one")
        self.assertIs(var_scope_one, x)

        # Q: What if we open scope_one within some other scope?
        # A: Identical behavior (in terms of names) as doing it outside.
        with tf.variable_scope("some_other_scope") as scope:
            self.assertEqual("some_other_scope", scope.name)
            with tf.variable_scope(scope_one):
                self.assertEqual("scope_level_1", tf.get_variable_scope().name)

        with tf.variable_scope("scopey_mc_scopeyface") as scope:
            for _ in range(2):
                self.assertTrue(scope.reuse == False)
                scope.reuse_variables()
            self.assertTrue(scope.reuse == True)

    def test_name_scope(self):
        """Name scope is for ops only. Explore relationship to variable_scope."""
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger('TestTFGeneral.test_name_scope')
        with tf.variable_scope("var_scope"):
            with tf.name_scope("name_scope"):
                var = tf.get_variable("var", [1])
                x   = 1.0 + var
        self.assertEqual("var_scope/var:0", var.name)
        self.assertEqual("var_scope/name_scope/add", x.op.name) # Ignore pycharm.
        log.info("\n\nx.op is %r" % x.op)
        log.info("\n\nvar.op is %r" % var.op)



    def test_get_variable(self):
        """Unclear what get_variable returns in certain situations. Want to explore."""
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger('TestTFGeneral.test_get_variable')

        # Returned object is a Tensor.
        unicorns = tf.get_variable("unicorns", [5, 7])
        log.info("\n\nUnicorns:\n\t{}".format(unicorns))
        log.info("\tName: {}".format(unicorns.name))
        self.assertEqual("unicorns:0", unicorns.name)

        with tf.variable_scope("uniscope") as scope:
            var = tf.get_variable("scoped_unicorn", [5, 7])
            log.info("Scoped unicorn name: {}".format(var.name))
            self.assertEqual("uniscope/scoped_unicorn:0", var.name)

            # What happens when we try to get_variable on previously created name?
            scope.reuse_variables() # MANDATORY
            varSame = tf.get_variable("scoped_unicorn")
            self.assertTrue(var is varSame)

    def test_tensors(self):
        """Tensor iteration is weird. Just kidding it's impossible. """
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger('TestTFGeneral.test_tensors')

        batch_size = 64
        num_inputs = 5
        tensor = tf.get_variable("2DTensor", shape=[batch_size, num_inputs])
        log.info("\n\n2D tensor:\n\t%r" % tensor)
        log.info("\tShape: %r" % tensor.get_shape())

        with self.assertRaises(TypeError):
            log.info("\n\n")
            for t in tensor:
                log.info("t: %r" % t)
                log.info("\tt shape: %r" % t.get_shape())






@unittest.skip("Methods used in this test may no longer exist. Revise or delete.")
class TestTensorboard(unittest.TestCase):

    # setUp is called by unittest before any/all test(s).
    def setUp(self):
        self.config = TrainConfig(FLAGS)
        buckets = [(5, 10)]
        self.bot = chatbot.Chatbot(buckets,
                              layer_size=FLAGS.layer_size,
                              num_layers=FLAGS.num_layers,
                              debug_mode=True)

    def test_train_step(self):
        """Check for expected outputs of chatbot.step."""

        def _get_data_distribution(train_set, buckets):
            train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
            train_total_size   = float(sum(train_bucket_sizes))
            return [sum(train_bucket_sizes[:i + 1]) / train_total_size
                             for i in range(len(train_bucket_sizes))]

        self.bot.setup_parameters(self.config)
        dataset = get_dataset(FLAGS.data_name, FLAGS.vocab_size)

        with self.bot.sess as sess:
            print ("Reading development and training data (limit: %d)." % self.config.max_train_samples)
            train_set, dev_set = read_data(dataset, self.bot.buckets, max_train_data_size=1e4)
            train_buckets_scale = _get_data_distribution(train_set, self.bot.buckets)
            step_time, loss = 0.0, 0.0
            previous_losses = []
            for i_step in range(5):
                rand = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = self.bot.get_batch(train_set, bucket_id)
                step_returns = self.bot.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,False)

                merged, gradient_norms, losses, _ = step_returns
                self.bot.file_writer.add_summary(merged, i_step)
                step_time += (time.time() - start_time) / self.config.steps_per_ckpt
                loss      += losses / self.config.steps_per_ckpt


if __name__ == '__main__':
    unittest.main()

