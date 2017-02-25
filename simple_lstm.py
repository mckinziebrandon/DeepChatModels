import tensorflow as tf
import numpy as np
import time
from util.printer import *
from util.reader import ptb_raw_data, ptb_producer
import pdb

# https://www.tensorflow.org/tutorials/recurrent
DATA_PATH='/home/brandon/terabyte/Datasets/simple-examples/data'
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


flags = tf.flags
flags.DEFINE_string("save_path", "./models/", "Model output directory.")
FLAGS=flags.FLAGS

class SimpleInput(object):
    def __init__(self, data, batch_size=32, num_steps=20, name=None):
        self.batch_size = batch_size
        self.num_steps = num_steps
        # TODO: weird, am i right?
        self.epoch_size = ((len(data) // self.batch_size) - 1) // self.num_steps
        self.input_data, self.targets = ptb_producer(data, self.batch_size, self.num_steps, name=name)

class SimpleModel(object):

    # Stuff removed:
    # - dropout layers
    # - attention
    def __init__(self, hidden_size, vocab_size, is_training, input_):

        # ==========================================================
        # All instance variables AND CONFIG VALUES here.
        # ==========================================================
        self._input = input_
        batch_size  = input_.batch_size
        num_steps   = input_.num_steps
        size        = hidden_size
        vocab_size  = vocab_size
        num_layers = 3 # config value
        max_grad_norm = 5 # config value

        def lstm(num_units=hidden_size):
            # BasicLSTMCell Path: tensorflow /contrib/rnn/python/ops/core_rnn_cell_impl.py.
            return tf.contrib.rnn.BasicLSTMCell(num_units=num_units,
                                                forget_bias=0.0,        # default: 1.0
                                                state_is_tuple=True)    # default: True


        cell = tf.contrib.rnn.MultiRNNCell([lstm() for _ in range(num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        embedding   = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
        inputs      = tf.nn.embedding_lookup(embedding, input_.input_data)
#
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=self._initial_state)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],
                                                                  [tf.reshape(input_.targets, [-1])],
                                                                  [tf.ones([batch_size * num_steps], dtype=tf.float32)])

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr        = tf.Variable(0.0, trainable=False)
        tvars           = tf.trainable_variables()
        grads, _        = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
        optimizer       = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op  = optimizer.apply_gradients(zip(grads, tvars),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32,
                                      shape=[],
                                      name="new_learning_rate")

        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op



if __name__ == "__main__":

    # Keep all training parameters here (if possible).
    batch_size = 32
    num_time_steps = 1

    # Setup data.
    train_data, valid_data, test_data, vocab_size = ptb_raw_data(DATA_PATH)
    describe_list("train_data", train_data)
    describe_list("valid_data", train_data)
    describe_list("test_data", test_data)
    print("vocabulary=", vocab_size)

    # y_train is just X_train time-shifted by one (think about it).
    # They both have shape [batch_size, num_steps]
    #X_train, y_train = ptb_producer(train_data, batch_size, num_steps=num_time_steps)
    #print("X_train.shape =", X_train.shape)
    #print("y_train.shape =", y_train.shape)


    # Setup model.
    #initial_state = tf.zeros(shape=[batch_size, model.state_size[1]])

    probabilities = []
    loss = 0.0

    with tf.Graph().as_default():

        initializer = tf.random_uniform_initializer(-0.1, 0.1)

        with tf.name_scope("Train"):
            train_input = SimpleInput(data=train_data)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = SimpleModel(hidden_size=200,
                                    vocab_size=vocab_size,
                                    is_training=True,
                                    input_=train_input)

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
                mtest = SimpleModel(hidden_size=200, vocab_size=vocab_size, is_training=False, input_=test_input)

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





