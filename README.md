# Conversation Models in Tensorflow

[NEW MODEL: DynamicBot. More info in next section]

This project is still very much evolving each day, but the core goals are:
* Create a cleaner user interface for tinkering with sequence-to-sequence models and over multiple datasets. This project will explore ways to make constructing such models feel more intuitive/customizeable. The ideal result is a chatbot API with the readability of [Keras](https://keras.io/), but with a degree of flexibility closer to TensorFlow. For example, the following code is all that is needed (after imports, etc.) to create and train one of the models on the Cornell movie dialogs:
```python
    # (Optional) Number of training samples used per gradient update.
    batch_size = 64
    # (Optional) Specify the max allowed number of words per sentence.
    max_seq_len = 500
    
    # All datasets implement a Dataset interface, found in data/_dataset.py
    dataset = Cornell(FLAGS.vocab_size)

    # Create chat model of choice. Pass in FLAGS values in case you want to change from defaults.
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     ckpt_dir=FLAGS.ckpt_dir,
                     batch_size=FLAGS.batch_size,
                     state_size=FLAGS.state_size,
                     embed_size=FLAGS.embed_size,
                     learning_rate=FLAGS.learning_rate,
                     lr_decay=FLAGS.lr_decay,
                     is_chatting=FLAGS.decode)


    # Don't forget to compile! Name inspired by Keras method of the same name.
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"
    if not FLAGS.decode:
        print("Training bot. CTRL-C to stop training.")
        bot.train(dataset.train_data, dataset.valid_data,
                  nb_epoch=FLAGS.nb_epoch,
                  steps_per_ckpt=FLAGS.steps_per_ckpt)

    else:
        print("Initiating chat session")
        bot.decode()
```

* Explore how [personalities of chatbots](https://arxiv.org/pdf/1603.06155.pdf) change when trained on different datasets, and methods for improving speaker consistency.
* Add support for "teacher mode": an interactive chat session where the user can tell the bot how well they're doing, and suggest better responses that the bot can learn from.

* [Completed] Models:
    * Rewrite conversation model with faster embedding technique and new TF support for dynamic unrolling.
    * Implement an attention-based embedding sequence-to-sequence model with the help of the tensorflow.contrib libraries.
    * Implement a simpler embedding sequence-to-sequence from "scratch" (minimal use of contrib).
* [Completed] Datasets:
    * **WMT'15** : English-to-French translation.
    * **Ubuntu Dialogue Corpus**: Reformatted as single-turn to single-response pairs.
    * **Cornell Movie-Dialogs**: Recently (March 5) incorporated [this preprocessed](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus) version of the Cornell corpus. I'll be processing and reformatting it further.

## Faster Embedding, Encoding, and Chatting

The newest model, ```DynamicBot```, is substantially faster than the previous models (bucketed models in legacy_models.py). Here are some of the key design differences for comparison:

|               | BucketModel | DynamicBot |
| ------------  | ----------    | -----------------------   |
| Embedding     | Used TensorFlow's ```EmbeddingWrapper```, which computes the embedding on a batch at each timestep. | Uses custom Embedder class to dynamically embed full batch-concatenated inputs of variable sequence length. |
| Encoding      | Employed the standard 'bucketed' model as described in TensorFlow sequence-to-sequence tutorial. Requires inputs to be padded to the same sequence length, for each bucket, which can result in unnecessarily large matrices of mainly zeros. | Combines the functionality of the new dynamic_rnn method in Tensorflow r1.0, wrapped inside a custom Encoder class. Input sequences are first fed to a custom batch_padded preprocessing utility (see utils/io_utils) that drastically reduces the occurrence of zero-padded sequences and allows for variable-length sequence batches. |
| Chatting      | Requires output to be assigned to a bucket, which constrains the raw output sequences to be constrained to pre-defined lengths. They then have to be truncated to remove padding. | Responses are generated naturally: once DynamicBot has read your input, it writes its response word by word until it signals that it's done speaking. No awkward post-processing required, and faster response times. |

One particular feature of DynamicBot worth mentioning is that the output generation and sampling process is _fully contained within the graph_ structure itself. This is in contrast with methods of outputting large arrays representing the logits (unnormalized log probabilities) and then sampling/argmax-ing over these. DynamicBot, however, directly returns its generated responses as a sequence of word-tokens.


## Sanity checks

Now that the goals for DynamicBot have been met design-wise, I'm digging into the first big testing/debugging stage.

### Check 1: Ensure a large DynamicBot can overfit a small dataset.

Below is a plot related to one of the debugging strategies recommended in chapter 11 of *Deep Learning* by Goodfellow et al. The idea is that any sufficiently large model should be able to perfectly fit (well, overfit) a small training dataset. I wanted to make sure DynamicBot could overfit before I started implementing any regularizing techniques. It is a plot in TensorBoard of cross-entropy loss (y-axis) against global training steps (x-axis). The orange curve is the training loss, while the blue curve is the validation loss. TensorBoard has visually smoothed out the oscillations a bit. 

![Ensuring DynamicBot can overfit before optimizing any further](http://i.imgur.com/PwhSmwJ.png)

This plot shows DynamicBot can achieve 0 loss for an extremely small dataset. Great, we can overfit. Now we can begin to explore regularization techniques.

### Check 2: Random & Grid Search Plots

I recently did a small random search and grid search over the following hyperparameters: learning rate, embed size, state size. The plots below show some of the findings. These are simply exploratory, I understand their limitations and I'm not drawing strong conclusions from them. They are meant to give a rough sense of the energy landscape in hyperparameter space. Oh and, plots make me happy. Enjoy. For all below, the y-axis is validation loss and the x-axis is global (training) step. The colors distinguish between model hyperparameters defined in the legends.





<img alt="state_size" src="http://i.imgur.com/w479tSo.png" width="400" align="left">
<img alt="embed_size" src="http://i.imgur.com/2Tj3vmA.png" width="400">
<br/>
<br/>


The only takeaway I saw from these two plots (after seeing the learning rate plots below) is that the __learning rate__, not the embed size, is overwhelmingly for responsible for any patterns here. It also looks like models with certain emed sizes (like 30) were underrepresented in the sampling, we see less points for them than others. The plots below illustrate the learning rate dependence.


<img alt="learning_rate" src="http://i.imgur.com/CtpX6vr.png" width="600">
<br/><br/>
<br/>
<br/>

Hmm, the wild oscillations for the large learning rate of 0.7 were expected, but what is going on with the values lying along the bottom (also with 0.7)? Perhaps we can find out by peering in on the same style of plot for each individual learning rate, as done below.

<img alt="learning_subs" src="http://i.imgur.com/bD8MFrV.png" width="900">

**General conclusion: the learning rate influences the validation loss far more than state size or embed size.** This was basically known before making these plots, as it is a well known property of such networks (Ng). It was nice to verify this for myself.


