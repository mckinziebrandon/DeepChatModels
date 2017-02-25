# Heavily uses code from: https://www.tensorflow.org/tutorials/recurrent
import tensorflow as tf
import numpy as np
import time
from util.printer import *
from util.reader import ptb_raw_data, ptb_producer
import pdb

flags = tf.flags
flags.DEFINE_string("save_path", "./models/", "Model output directory.")
FLAGS=flags.FLAGS

DATA_PATH='/home/brandon/terabyte/Datasets/simple-examples/data'

# Global hyperparameters.
PARAMS = {'learning_rate': 1.0,
          'max_grad_norm': 5,
          'num_layers': 2,
          'num_steps': 20,
          'hidden_size': 200,
          'max_epoch': 4,
          'max_max_epoch': 13,
          'keep_prob': 1.0,
          'lr_decay': 0.5,
          'batch_size': 20,
          'vocab_size': 10000}


def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)



class SimpleInput(object):
    def __init__(self, data, name=None):
        self.epoch_size = ((len(data) // PARAMS['batch_size'] - 1) // PARAMS['num_steps']
        self.input_data, self.targets = ptb_producer(data, PARAMS['batch_size'], PARAMS['num_steps'], name=name)

class SimpleModel(object):

    def __init__(self, is_training, input):
        # Imports drastically improve readability, as opposed to retyping tf.contrib.... everywhere.
        # It also allows PyCharm to give descriptive hover-documentation.
        from tensorflow.contrib.rnn import MultiRNNCell, static_rnn, BasicLSTMCell
        from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
        self._input = input

        # ==========================================================
        # Define the model architecture.
        # ==========================================================

        # The recurrent components.
        def lstm(): return BasicLSTMCell(num_units=PARAMS['hidden_size'], forget_bias=0.0)
        cell = MultiRNNCell([lstm() for _ in range(PARAMS['num_layers'])])
        self._initial_state = cell.zero_state(batch_size, tf.float32)           # "return zero-filled state tensors."

        # The input components.
        embedding       = tf.get_variable("embedding", [PARAMS['vocab_size'], PARAMS['hidden_size']], dtype=tf.float32)
        inputs          = tf.nn.embedding_lookup(embedding, input_.input_data)
        inputs          = tf.unstack(inputs, num=PARAMS['num_steps'], axis=1)

        # The output components.
        outputs, state  = static_rnn(cell, inputs, initial_state=self._initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, PARAMS['hidden_size']])
        softmax_w = tf.get_variable("softmax_w", [PARAMS['hidden_size'], PARAMS['vocab_size']], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [PARAMS['vocab_size']], dtype=tf.float32)


        # ==========================================================
        # Configure the learning process.
        # ==========================================================

        # Choose logistic loss for training function.
        logits  = tf.matmul(output, softmax_w) + softmax_b
        loss    = sequence_loss_by_example(logits=[logits],
                                           targets=[tf.reshape(input_.targets, [-1])],
                                           weights=[tf.ones([batch_size * PARAMS['num_steps']], dtype=tf.float32)])

        self.cost           = tf.reduce_sum(loss) / batch_size
        self.final_state    = state
        if not is_training: return

        # TODO: Why is this set to zero?
        self.learning_rate  = tf.Variable(initial_value=0.0, trainable=False)
        grads, _            = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), max_grad_norm)
        optimizer           = tf.train.GradientDescentOptimizer(self.learning_rate)
        self._train_op  = optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())

        # wth is this code design
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.learning_rate, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


if __name__ == "__main__":

    # Keep all training parameters here (if possible).
    batch_size = 32
    num_time_steps = 1

    # ===========================================================================================
    # Setup
    # ===========================================================================================

    # Get data in the form of integer lists.
    # The integers are indices to a vocabulary (size 'vocab_size') of words.
    train_data, valid_data, test_data, _ = ptb_raw_data(DATA_PATH)
    describe_list("train_data", train_data)
    describe_list("valid_data", train_data)
    describe_list("test_data", test_data)
    print("vocabulary=", PARAMS['vocab_size'])


    # Open tf.Graph.as_default context manager, which overrides the current default graph for the lifetime of the context.
    with tf.Graph().as_default():

        # random_uniform_initializer generates tensors with a uniform distribution over the specified range.
        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        def createModel(name_scope, data, is_training=False, reuse=True):
            with tf.name_scope(name_scope):
                input = SimpleInput(data=data, name=name_scope+"Input")
                with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
                    model = SimpleModel(is_training=is_training, input=input)
                if name_scope == "Train":
                    tf.summary.scalar("Training Loss", model.cost)
                    tf.summary.scalar("Learning Rate", model.learning_rate)
                elif name_scope == "Valid":
                    tf.summary.scalar("Validation Loss", model.cost)





        with tf.name_scope("Train"):
            train_input = SimpleInput(data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = SimpleModel(is_training=True, input_=train_input)

                tf.summary.scalar("Training Loss", model.cost)
                tf.summary.scalar("Learning Rate", model.lr)

        with tf.name_scope("Valid"):
            valid_input = SimpleInput(data=valid_data, name="ValidInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mvalid = SimpleModel(hidden_size=200, vocab_size=vocab_size, is_training=False, input_=valid_input)
            tf.summary.scalar("Validation Loss", mvalid.cost)


        with tf.name_scope("Test"):
            test_input = SimpleInput(data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                mtest = SimpleModel(is_training=False, input_=test_input)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)

        print("giddyup!!!")
        max_max_epoch = 13
        _max_epoch = 4
        _lr_decay = 0.5
        _learning_rate = 1.0
        with sv.managed_session() as session:
            for i in range(max_max_epoch):
                # wtf
                lr_decay = _lr_decay ** max(i + 1 - _max_epoch, 0.0)
                model.assign_lr(session, _learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(model.lr)))
                train_perplexity = run_epoch(session, model, eval_op=model._train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, mvalid)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)





