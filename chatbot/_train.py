"""Train seq2seq attention chatbot on ubuntu dialogue corpus."""
import time
import numpy as np
from utils import *



def _train(chatbot, num_steps=100000):
    """ Train chatbot using 1-on-1 ubuntu dialogue corpus. """

    with chatbot.sess as sess:
        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % chatbot.config.max_train_samples)
        train_set, dev_set = data_utils.read_data("ubuntu",
                                                  chatbot.config.data_dir,
                                                  chatbot.buckets,
                                                  chatbot.vocab_size,
                                                  max_train_data_size=chatbot.config.max_train_samples)

        # Get number of samples for each bucket (i.e. train_bucket_sizes[1] == num-trn-samples-in-bucket-1).
        train_bucket_sizes = [len(train_set[b]) for b in range(len(chatbot.buckets))]
        # The total number training samples, excluding the ones too long for our bucket choices.
        train_total_size   = float(sum(train_bucket_sizes))

        # Interpret as: train_buckets_scale[i] == [cumulative] fraction of samples in bucket i or below.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        previous_losses = []
        for i_step in range(num_steps):
            # Sample a random bucket index according to the data distribution,
            # then get a batch of data from that bucket by calling chatbot.get_batch.
            rand = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > rand])

            # Get a batch and make a step.
            start_time = time.time()
            # Recall that target_weights are NOT parameter weights; they are weights in the sense of "weighted average."
            encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch(train_set, bucket_id)

            _, avg_perplexity, _ = chatbot.step(sess,
                                         encoder_inputs,
                                         decoder_inputs,
                                         target_weights,
                                         bucket_id,
                                         False)

            step_time += (time.time() - start_time) / chatbot.config.steps_per_ckpt
            loss      += avg_perplexity / chatbot.config.steps_per_ckpt

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if i_step % chatbot.config.steps_per_ckpt == 0:
                # Print statistics for the previous epoch.
                perplexity = np.exp(float(loss)) if loss < 300 else float("inf")
                print("\nglobal step:", chatbot.global_step.eval(), end="  ")
                print("learning rate: %.4f" %  chatbot.learning_rate.eval(), end="  ")
                print("step time: %.2f" % step_time, end="  ")
                print("perplexity: %.2f" % perplexity)

                # Decrease learning rate more aggressively.
                if len(previous_losses) > 3 and loss > min(previous_losses[-3:]):
                    sess.run(chatbot.lr_decay_op)
                previous_losses.append(loss)

                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(chatbot.config.ckpt_dir, "converse.ckpt")
                chatbot.saver.save(sess, checkpoint_path, global_step=chatbot.global_step)
                step_time, loss = 0.0, 0.0

                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(chatbot.buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = chatbot.get_batch(
                    dev_set, bucket_id)
                    _, eval_loss, _ = chatbot.step(sess, encoder_inputs, decoder_inputs,
                    target_weights, bucket_id, True)
                    eval_ppx = np.exp(float(eval_loss)) if eval_loss < 300 else float(
                    "inf")
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

        checkpoint_path = os.path.join(chatbot.config.ckpt_dir, "converse.ckpt")
        chatbot.saver.save(sess, checkpoint_path, global_step=chatbot.global_step)

