import logging
import sys
sys.path.append("..")
import unittest
import tensorflow as tf
from utils import io_utils
from utils import bot_freezer

test_flags = tf.app.flags
test_flags.DEFINE_string("config", "configs/test_config.yml", "path to config (.yml) file.")
test_flags.DEFINE_string("model", "{}", "Options: chatbot.{DynamicBot,Simplebot,ChatBot}.")
test_flags.DEFINE_string("model_params", "{}", "")
test_flags.DEFINE_string("dataset", "{}", "Options: data.{Cornell,Ubuntu,WMT}.")
test_flags.DEFINE_string("dataset_params", "{}", "")
TEST_FLAGS = test_flags.FLAGS


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


class TestGraphOps(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger('TestGraphOps')
        self._build_simple_graph()

    def _build_simple_graph(self):

        self.batch_size = 32
        self.state_size = 512
        self.vocab_size = 1000
        self.x = tf.placeholder(dtype=tf.int32,
                                shape=(self.batch_size, self.state_size),
                                name="x")
        self.W = tf.get_variable(name="W",
                                 shape=(self.state_size, self.vocab_size),
                                 dtype=tf.float32)
        self.h = tf.matmul(tf.cast(self.x, tf.float32), self.W)
        self.y = tf.add(self.h, 2., name="y")

    def _print_op_names(self, g):
        print("List of Graph Ops:")
        for op in g.get_operations():
            print(op.name)

    def test_collections(self):
        g = tf.get_default_graph()
        self._print_op_names(g)

        collection_name = "test_coll"
        print("Adding x and y to collection %s" % collection_name)
        tf.add_to_collection(collection_name, self.x)
        tf.add_to_collection(collection_name, self.y)

        print("All collection keys:")
        print(g.get_all_collection_keys())

        print("tf.get_collection({}) = {}".format(
            collection_name, tf.get_collection(collection_name)
        ))


    def test_load(self):

        config = io_utils.parse_config(TEST_FLAGS)
        frozen_graph = bot_freezer.load_graph(config['model_params']['ckpt_dir'])
        self._print_op_names(frozen_graph)

        print("frozen_graph:\n", frozen_graph)
        print("frozen_graph.get_all_collection_keys() =",
              frozen_graph.get_all_collection_keys())


if __name__ == '__main__':
    unittest.main()

