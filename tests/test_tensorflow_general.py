import logging
import sys
sys.path.append("..")
import os
import unittest
import tensorflow as tf
import numpy as np
import time
from utils.io_utils import *
from chatbot import DynamicBot
from data import Cornell, Ubuntu, TestData

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

TEMP="/home/brandon/terabyte/Datasets/ubuntu_dialogue_corpus"
BASE='/home/brandon/Documents/seq2seq_projects/tests'


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


    def test_input_pipeline(self):
        """First steps at the confusing input pipeline. """

        n_samples = 6 # of one datapoint.
        capacity = 3 # length of queue

        x = tf.random_normal([n_samples], mean=-1, stddev=4)

        # Create a waiting line.
        q = tf.FIFOQueue(capacity=capacity, dtypes=tf.float32)

        # To check what is happening in this case:
        # we will print a message each time "x_input_data" is actually computed
        # to be used in the "enqueue_many" operation
        x = tf.Print(x, data=[x], message="Raw inputs data generated:", summarize=n_samples)
        enqueue_op = q.enqueue_many(x)

        # We need an operation that will actually fill the queue with our data
        # "enqueue_many" slices "x" along the 0th dimension to make multiple queue elements
        #enqueue_op = q.enqueue_many(x)  # <- x1 - x2 -x3 |

        # To leverage multi-threading we create a "QueueRunner"
        # that will handle the "enqueue_op" outside of the main thread
        # We don't need much parallelism here, so we will use only 1 thread
        numberOfThreads = 1
        qr = tf.train.QueueRunner(q, [enqueue_op] * numberOfThreads)
        # Don't forget to add your "QueueRunner" to the QUEUE_RUNNERS collection
        tf.train.add_queue_runner(qr)

        # We need a dequeue op to get the next elements in the queue following the FIFO policy.
        input = q.dequeue()
        # The input tensor is the equivalent of a PLACEHOLDER now,
        # but directly connected to the data sources in the graph
        # Each time we use the input tensor, we print the number of elements left
        # in the queue
        input = tf.Print(input, data=[q.size(), input], message="Nb elements left:")

        # fake graph: START
        y = input + 1
        # fake graph: END

        with tf.Session() as sess:
            # But now we build our coordinator to coordinate our child threads with
            # the main thread
            coord = tf.train.Coordinator()
            # Beware, if you don't start all your queues before runnig anything
            # The main threads will wait for them to start and you will hang again
            # This helper start all queues in tf.GraphKeys.QUEUE_RUNNERS
            threads = tf.train.start_queue_runners(coord=coord)

            # The QueueRunner will automatically call the enqueue operation
            # asynchronously in its own thread ensuring that the queue is always full
            # No more hanging for the main process, no more waiting for the GPU
            sess.run(y)
            sess.run(y)

            # We request our child threads to stop ...
            coord.request_stop()
            # ... and we wait for them to do so before releasing the main thread
            coord.join(threads)


class TestRNN(unittest.TestCase):
    """Test behavior of tf.contrib.rnn after migrating to r1.0."""

    def setUp(self):
        self.batch_size = 32
        self.input_size = 1000
        self.seq_len = 20
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestRNNLogger')

    def test_static_rnn(self):
        self.log.info("\n")

        with tf.variable_scope("inputs"):
            inp_name = "static_rnn_input"
            inp_shape = [self.batch_size, self.input_size]
            inputs = [tf.placeholder(tf.float32, inp_shape, inp_name+str(i))
                           for i in range(self.seq_len)]

        with tf.variable_scope("encoder"):
            basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=128)
            # static_rnn creates the network and returns the list of outputs at each step,
            # and the final state in Tensor form.
            encoder_outputs, encoder_state = tf.contrib.rnn.static_rnn(basic_cell,
                                                                       inputs,
                                                                       dtype=tf.float32)
            self.log.info("\nType information:")
            self.log.info("\ttype(encoder_outputs): {}".format(type(encoder_outputs)))
            self.log.info("\ttype(encoder_state): {}".format(type(encoder_state)))

    def test_dynamic_rnn(self):
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=128)

        inputs = tf.placeholder(tf.float32, [self.batch_size, self.seq_len, self.input_size])
        # "fully dynamic unrolling of inputs."
        outputs, state = tf.nn.dynamic_rnn(basic_cell,
                                           inputs,
                                           dtype=tf.float32)


class TestTensorflowSaver(unittest.TestCase):

    def test_simple_save(self):

        # Create some simple variables
        w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
        w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')

        # Add them to collection.
        tf.add_to_collection(name='test_simple_save', value=w1)
        tf.add_to_collection(name='test_simple_save', value=w2)

        # Create object that saves stuff.
        saver = tf.train.Saver()

        # Run the session; populates variables with vals.
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # How to save.
        saved_path = saver.save(sess, 'out/model_test_simple_save')
        self.assertTrue(saved_path != None)

    def test_restore(self):
        """Ensures we can load the full model from test_simple_save."""
        sess = tf.Session()

        new_saver = tf.train.import_meta_graph('out/model_test_simple_save.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('out/'))

        all_vars = tf.get_collection('test_simple_save')
        for v in all_vars:
            self.assertTrue(type(v) == type(tf.Variable(0)))
            v_ = sess.run(v)
            print(v_)

    def test_save_placeholder(self):

        place_holder = tf.placeholder(tf.float32, name="test_ph")
        tf.add_to_collection(name="test_save_placeholder", value=place_holder)

        place_holder_2 = tf.placeholder(tf.float32, name="test_ph_2")
        tf.add_to_collection(name="test_save_placeholder", value=[place_holder,place_holder_2])


    def test_freezer(self):
        """Simple start-to-finish testing of tensorflow model freezing."""

        batch_size = 32
        state_size = 512
        vocab_size = 1000
        inputs = tf.placeholder(dtype=tf.int32, shape=(batch_size, state_size))
        h = tf.add(inputs, 42)
        y = tf.multiply(h, 3)


        with tf.Session() as sess:
            x = np.random.randint(vocab_size, size=(batch_size, state_size))
            print(sess.run(y, {inputs: x}))


if __name__ == '__main__':
    unittest.main()














































