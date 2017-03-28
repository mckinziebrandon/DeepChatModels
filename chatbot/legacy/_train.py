"""Train seq2seq attention chatbot.
Note: Only used for legacy_models.
For (better) DynamicBot implementation, please see dynamic_models.py and, for saving/restoring ops,
the base class of all models in _models.py.
"""
import time
from utils import *

def train(bot, dataset):
    """ Train chatbot using dataset given by dataset.
        chatbot: instance of ChatBot or SimpleBot.
    """

    # Get data as token-ids.
    train_set, dev_set = io_utils.read_data(dataset,
                                            bot.buckets)

    # Interpret train_buckets_scale[i] as [cumulative] frac of samples in bucket i or below.
    train_buckets_scale = _get_data_distribution(train_set, bot.buckets)

    # This is the training loop.
    i_step = 0
    step_time, loss = 0.0, 0.0
    previous_losses = []
    try:
        while True:
            # Sample a random bucket index according to the data distribution,
            # then get a batch of data from that bucket by calling chatbot.get_batch.
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            # Get a batch and make a step.
            start_time = time.time()
            summary, step_loss = run_train_step(bot, train_set, bucket_id, False)
            step_time += (time.time() - start_time) / bot.steps_per_ckpt
            loss      += step_loss / bot.steps_per_ckpt

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % bot.steps_per_ckpt == 0:
                run_checkpoint(bot, step_time, loss, previous_losses, dev_set)
                step_time, loss = 0.0, 0.0
            i_step += 1
    except (KeyboardInterrupt, SystemExit):
        print("Training halted. Cleaning up . . . ")
        # Store the model's graph in ckpt directory.
        bot.saver.export_meta_graph(bot.ckpt_dir + dataset.name + '.meta')
        bot.close()
        print("Done.")


def run_train_step(model, train_set, bucket_id, forward_only=False):
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
    step_returns = model.step(encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only)
    summary, _, losses, _ = step_returns
    if not forward_only and summary is not None:
        model.train_writer.add_summary(summary, model.global_step.eval(model.sess))
    return summary, losses


def run_checkpoint(model, step_time, loss, previous_losses, dev_set):
    # Print statistics for the previous epoch.
    perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
    print("\nglobal step:", model.global_step.eval(model.sess), end="  ")
    print("learning rate: %.4f" %  model.learning_rate.eval(session=model.sess), end="  ")
    print("step time: %.2f" % step_time, end="  ")
    print("perplexity: %.2f" % perplexity)

    # Run evals on development set and print their perplexity.
    for bucket_id in range(len(model.buckets)):
        if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
        summary, eval_loss = run_train_step(model, dev_set, bucket_id, forward_only=True)
        model.save(summaries=summary)
        eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
    sys.stdout.flush()


def _get_data_distribution(train_set, buckets):
    # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
    train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
    # The total number training samples, excluding the ones too long for our bucket choices.
    train_total_size   = float(sum(train_bucket_sizes))

    # Interpret as: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
    return [sum(train_bucket_sizes[:i + 1]) / train_total_size
                     for i in range(len(train_bucket_sizes))]


