"""Train seq2seq attention chatbot."""
import time
import numpy as np
from utils import *


def _train(chatbot, config):
    """ Train chatbot using dataset given by config.dataset. """

    # Number of unique training samples to consider for training.
    num_samples_total       = config.max_train_samples
    # User-specified choice of how many training samples they can comfortably store in RAM memory.
    # If this is less than the total, the training will be 'chunked' in portions of this size.
    samples_per_chunk       = config.chunk_size
    steps_per_chunk         = samples_per_chunk // chatbot.batch_size
    # Compute number of chunks.
    # Due to integer division, up to num_lines_per_chunk - 1 samples may not get used.
    num_chunks = num_samples_total // samples_per_chunk

    current_chunk = _get_current_chunk(chatbot, config, steps_per_chunk)
    if chatbot.debug_mode:
        print("[DEBUG] ========== Chunking information: ================")
        print("[DEBUG] Starting at chunk", current_chunk, "out of", num_chunks, "total.")
        print("[DEBUG] Steps per chunk:", steps_per_chunk)

    with chatbot.sess as sess:
        total_steps = 0
        for chunk in range(current_chunk, num_chunks):
            print("CHUNK NUMBER ", chunk)
            # Read data into buckets and compute their sizes.
            print ("Reading development and training data (limit: %d)." % config.max_train_samples)
            train_set, dev_set = data_utils.read_data(config.dataset,
                                                      chatbot.buckets,
                                                      max_train_data_size=samples_per_chunk,
                                                      skip_first=chunk*samples_per_chunk)

            # Interpret as: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
            train_buckets_scale = _get_data_distribution(train_set, chatbot.buckets)

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            previous_losses = []
            for i_step in range(steps_per_chunk):
                # Sample a random bucket index according to the data distribution,
                # then get a batch of data from that bucket by calling chatbot.get_batch.
                rand = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

                # Get a batch and make a step.
                start_time = time.time()
                avg_perplexity = _step(sess, chatbot, train_set, bucket_id)
                total_steps += 1
                step_time += (time.time() - start_time) / config.steps_per_ckpt
                loss      += avg_perplexity / config.steps_per_ckpt

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if total_steps % config.steps_per_ckpt == 0:
                    _run_checkpoint(sess, chatbot, config, step_time, loss, previous_losses, dev_set)
                    step_time, loss = 0.0, 0.0


def _step(sess, model, train_set, bucket_id):
    # Recall that target_weights are NOT parameter weights; they are weights in the sense of "weighted average."
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)

    _, avg_perplexity, _ = model.step(sess,
                                 encoder_inputs,
                                 decoder_inputs,
                                 target_weights,
                                 bucket_id,
                                 False)
    return avg_perplexity

def _run_checkpoint(sess, model, config, step_time, loss, previous_losses, dev_set):
    # Print statistics for the previous epoch.
    perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
    print("\nglobal step:", model.global_step.eval(), end="  ")
    print("learning rate: %.4f" %  model.learning_rate.eval(), end="  ")
    print("step time: %.2f" % step_time, end="  ")
    print("perplexity: %.2f" % perplexity)

    # Decrease learning rate more aggressively.
    if len(previous_losses) > 3 and loss > min(previous_losses[-3:]):
        sess.run(model.lr_decay_op)
    previous_losses.append(loss)

    # Save checkpoint and zero timer and loss.
    checkpoint_path = os.path.join(config.ckpt_dir, "{}.ckpt".format(config.data_name))
    # Saves the state of all global variables.
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    # Run evals on development set and print their perplexity.
    for bucket_id in range(len(model.buckets)):
        if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        dev_set, bucket_id)
        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
        target_weights, bucket_id, True)
        eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float(
        "inf")
        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
    sys.stdout.flush()

def _get_current_chunk(model, config, steps_per_chunk):
    """Returns the chunk number to resume training on.
    """
    checkpoint_state  = tf.train.get_checkpoint_state(config.ckpt_dir)
    if not checkpoint_state or config.reset_model: return 0
    checkpoint_file  = checkpoint_state.model_checkpoint_path
    if model.debug_mode:
        print("[DEBUG] Getting chunk info from ckpt file: ", checkpoint_file)

    global_step =""
    started = False
    rel_path_start = checkpoint_file.find(config.data_name)
    for ch in checkpoint_file[rel_path_start:]:
        if started and not ch.isdigit(): break
        if ch.isdigit():
            started=True
            global_step += str(ch)

    global_step = int(global_step)
    return global_step // steps_per_chunk

def _get_data_distribution(train_set, buckets):
    # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
    train_bucket_sizes = [len(train_set[b]) for b in range(len(buckets))]
    # The total number training samples, excluding the ones too long for our bucket choices.
    train_total_size   = float(sum(train_bucket_sizes))

    # Interpret as: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
    return [sum(train_bucket_sizes[:i + 1]) / train_total_size
                     for i in range(len(train_bucket_sizes))]

