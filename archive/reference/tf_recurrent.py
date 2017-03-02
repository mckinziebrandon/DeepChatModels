# Misc. notes & reminders:
# - Heavily uses code from: https://www.tensorflow.org/tutorials/recurrent
# - Any mention of the 'm' state in TF documentation is the hidden state.
# - c.shape == h.shape by definition of LSTMs.
import tensorflow as tf
import numpy as np
import time
from utils.printer import *
from utils.reader import ptb_raw_data, ptb_producer
import pdb

flags = tf.flags
flags.DEFINE_string("save_path", "./out/", "Model output directory.")
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


class SimpleInput(object):
    def __init__(self, data, name=None):
        self.epoch_size = (len(data) // PARAMS['batch_size'] - 1) // PARAMS['num_steps']
        self.input_data, self.targets = ptb_producer(data, PARAMS['batch_size'], PARAMS['num_steps'], name=name)

class SimpleModel(object):

    def __init__(self, is_training, input_):
        # Imports drastically improve readability, as opposed to retyping tf.contrib.... everywhere.
        # It also allows PyCharm to give descriptive hover-documentation.
        from tensorflow.contrib.rnn import MultiRNNCell, static_rnn, BasicLSTMCell
        from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
        self._input = input_

        # ==========================================================
        # Define the model architecture.
        # ==========================================================

        # The recurrent components.
        def lstm(): return BasicLSTMCell(num_units=PARAMS['hidden_size'], forget_bias=0.0, state_is_tuple=True)
        # Important note: cell.state_size = (LSTMStateTuple(c=hid_size, h=hid_size) for _ in range(num_layers))
        cell = MultiRNNCell([lstm() for _ in range(PARAMS['num_layers'])])
        self._initial_state = cell.zero_state(PARAMS['batch_size'], tf.float32)           # "return zero-filled state tensors."

        # The input components.
        embedding       = tf.get_variable("embedding", [PARAMS['vocab_size'], PARAMS['hidden_size']], dtype=tf.float32)
        # The next line is where we actually specify to TF which data this model is fed.
        inputs          = tf.nn.embedding_lookup(embedding, input_.input_data)
        inputs          = tf.unstack(inputs, num=PARAMS['num_steps'], axis=1)

        # The output components.
        #pdb.set_trace()
        outputs, state  = static_rnn(cell, inputs, initial_state=self._initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, PARAMS['hidden_size']])
        softmax_w = tf.get_variable("softmax_w", [PARAMS['hidden_size'], PARAMS['vocab_size']], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [PARAMS['vocab_size']], dtype=tf.float32)


        # ==========================================================
        # Configure evaluation operations.
        # ==========================================================

        # Choose logistic loss for training function.
        logits  = tf.matmul(output, softmax_w) + softmax_b
        loss    = sequence_loss_by_example(logits=[logits],
                                           targets=[tf.reshape(self.input.targets, [-1])],
                                           weights=[tf.ones([PARAMS['batch_size'] * PARAMS['num_steps']], dtype=tf.float32)])

        self._cost           = tf.reduce_sum(loss) / PARAMS['batch_size']
        self._final_state    = state
        if not is_training: return

        # ==========================================================
        # Configure training operations (if is_training).
        # ==========================================================

        # Initialize learning rate to zero. It will be set before every training epoch.
        self._lr    = tf.Variable(initial_value=0.0, trainable=False)
        grads, _    = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), PARAMS['max_grad_norm'])
        optimizer   = tf.train.GradientDescentOptimizer(self._lr)
        # train_op stores the operation that actually updates the (trainable) params.
        self._train_op  = optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())

        # Define placeholder that will be container for new learning rate values at each new epoch.
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        # Define operation for reassigning contents of _new_lr to contents of _lr.
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    # Question: Why is it necessary to define these as properties in order for this to run correctly?
    # Answer: https://docs.python.org/3/library/functions.html#property
    # TL;DR: @property should really be called @getter_read_only
    # It is safer this way because the initial values are fed to the graph model, which would not know if we change them later.
    # Keep it this way or you will make me sad.
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
    def learning_rate(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


def run_epoch(session, model, train_op=None, verbose=False):
    """Runs model using session, storing state and cost at each epoch.

    Returns:
        perplexity, defined as exp(total_cost / num_iterations)
    """
    # TODO: should probably change eval_op to 'train_op' because obvious reasons.
    start_time = time.time()
    total_cost = 0.0
    iters = 0

    # Populate the initial state with zeros.
    state = session.run(model.initial_state)

    # Define the values we want to be returned after each session.run iteration.
    fetches = {"cost": model.cost, "final_state": model.final_state}
    if train_op is not None: fetches["eval_op"] = train_op

    for step in range(model.input.epoch_size):

        # The feed_dict is what tells the tf graph what real values to assign to the model components.
        # At the first step, feed dict will of course assign all zeros, since that is what running initial_state yields.
        # After that, 'state' contains the output after running the data through the graph at the previous step.
        feed_dict = {}
        # Loop over the stacked LSTMs. (num_layers iterations total).
        # This is how we reformat the previous outputs to be fed in again during this step.
        for i_LSTM_layer, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i_LSTM_layer].c
            feed_dict[h] = state[i_LSTM_layer].h


        # =======================================================================
        # Run the graph and tell us the cost and final state output.
        # =======================================================================

        fetches_evaluated = session.run(fetches, feed_dict)
        cost = fetches_evaluated["cost"]
        state = fetches_evaluated["final_state"]


        total_cost  += cost
        iters       += PARAMS['num_steps']

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(total_cost / iters),
                   iters * PARAMS['batch_size'] / (time.time() - start_time)))

    return np.exp(total_cost / iters)



if __name__ == "__main__":

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

        def createModel(name_scope, data, reuse=None, is_training=False):
            with tf.name_scope(name_scope):
                input = SimpleInput(data=data, name=name_scope+"Input")
                with tf.variable_scope("Model", reuse=reuse, initializer=initializer):
                    model = SimpleModel(is_training=is_training, input_=input)
                if name_scope == "Train":
                    tf.summary.scalar("Training Loss", model.cost)
                    tf.summary.scalar("Learning Rate", model.learning_rate)
                elif name_scope == "Valid":
                    tf.summary.scalar("Validation Loss", model.cost)
            return model

        # IMPORTANT: All of the following three are technically the same model,
        # but they differ in (1) the inputs they're fed, and (2) the summaries associated with them.
        train_model = createModel("Train", train_data, is_training=True)
        valid_model = createModel("Valid", valid_data, reuse=True)
        test_model  = createModel("Test",  test_data,  reuse=True)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as session:
            for i in range(PARAMS['max_max_epoch']):

                # Update the learning rate each epoch.
                learning_rate  = PARAMS['learning_rate']
                learning_rate *= PARAMS['lr_decay'] ** max(0.0, i - PARAMS['max_epoch'] + 1)
                train_model.assign_lr(session, learning_rate)
                print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model.learning_rate)))

                # Run training epoch.
                train_perplexity = run_epoch(session, train_model, train_op=train_model.train_op, verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))

                # Run validation epoch.
                valid_perplexity = run_epoch(session, valid_model)
                print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

            # Test our model.
            test_perplexity = run_epoch(session, test_model)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)





