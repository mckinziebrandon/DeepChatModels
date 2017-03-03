import os
import unittest
import tensorflow as tf

# --------  Notes: unittest module. --------
# Regarding this file:
# - TestTensorflowSaver is a single test case.
# - Any methods of it beginning with 'test' will be used by the test runner.
# - To take advantage of the testrunner, use self.assert{Equals, True, etc..} methods.
#
# More general:
# - @unittest.skip(msg) before a method will skip it.
# - right clicking inside the method and running DOES run that test only!

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


if __name__ == '__main__':
    unittest.main()


