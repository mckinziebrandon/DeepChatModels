# Conversation Models in Tensorflow

This project is still very much evolving each day, but the core goals are:
* Create a cleaner user interface for tinkering with sequence-to-sequence models. This project will explore ways to make constructing such models feel more intuitive/customizable. The ideal result is a chatbot API with the readability of [Keras](https://keras.io/), but with a degree of flexibility closer to [TensorFlow](https://www.tensorflow.org/). For example, the following code is all that is needed (after imports, etc.) to create and train one of the models on the Cornell movie dialogs (All params with '=' are optional) :
```python
    # All datasets implement a Dataset interface, found in data/_dataset.py
    dataset = Cornell(vocab_size=FLAGS.vocab_size)

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

    # Don't forget to compile! Name inspired by similar Keras method.
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
* Implement and improve "teacher mode": an interactive chat session where the user can tell the bot how well they're doing, and suggest better responses that the bot can learn from.

### Brief Overview of Completed Work


__Encoder/Decoder-Based Models__:
* DynamicBot: uses a more object-oriented approach offered by custom classes in model_components.py. The result is faster online batch-concatenated embedding and a more natural approach to chatting. It makes use of the (fantastic) new python API in the TensorFlow 1.0 release, notably the dynamic_rnn. It also adheres to good variable scoping practice and common tensorflow conventions I've observed in the documentation and source code, which has nice side effects such as clean graph visualizations in TensorBoard.

* SimpleBot: Simplified bucketed model based on the more complicated 'ChatBot' model below. Although it is less flexible in customizing bucket partitions and uses a sparse softmax over the full vocabulary instead of sampling, it is far more transparent in its implementation. It makes minimal use of tf.contrib, as opposed to ChatBot, and more or less is implemented from "scratch," in the sense of primarily relying on the basic tensorflow methods. If you're new to TensorFlow, it may be useful to read through its implementation to get a feel for common conventions in tensorflow programming, as it was the result of me reading the source code of all methods in ChatBot and writing my own more compact interpretation.

* ChatBot: Extended version of the model described in [this TensorFlow tutorial](https://www.tensorflow.org/tutorials/seq2seq). Architecture characteristics: bucketed inputs, decoder uses an attention mechanism (see page 52 of my [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf), and inputs are embedded with the simple functions provided in the tf.contrib library. Also employs a sampled softmax loss function to allow for larger vocabulary sizes (page 54 of [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf)). Additional comments: due to the nature of bucketed models, it takes much longer to create the model compared to others. The main bottleneck appears to be the size of the largest bucket and how the gradient ops are created based on the bucket sizes.


__Datasets__:
* [WMT'15](http://www.statmt.org/wmt15/translation-task.html): 22M sentences examples of english-to-french translation.

* [Ubuntu Dialogue Corpus](https://arxiv.org/pdf/1506.08909.pdf): pre-processing approach can be seen in the ubuntu\_reformat.ipynb in the notebooks folder. The intended use for the dataset is response ranking for multi-turn dialogues, but I've taken the rather simple approach of extracting utterance-pairs and interpreting them as single-sentence to single-response, which correspond with inputs for the encoder and decoder, respectively, in the models.

* [Cornell Movie-Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html): I began with [this preprocessed](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus) version of the Cornell corpus, and made minor modifications to reduce noise.

__[Ongoing] Reference Material__: A lot of research has gone into these models, and I've been documenting my notes on the most "important" papers here in the last section of [my deep learning notes here](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf). I'll be updating that as the ideas from more papers make their way into this project. The following is a list of the most influential papers for the architecture approaches in no particular order.
* [Sequence to Sequence Learning with Neural Networks. Sutskever et al., 2014.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
* [On Using Very Large Target Vocabulary for Neural Machine Translation. Jean et al., 2014.](https://arxiv.org/pdf/1412.2007.pdf)
* [Neural Machine Translation by Jointly Learning to Align and Translate. Bahdanau et al., 2016](https://arxiv.org/pdf/1409.0473.pdf)
* [Effective Approaches to Attention-based Neural Machine Translation. Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)


## Faster Embedding, Encoding, and Chatting

The newest model, ```DynamicBot```, is substantially faster than the previous models (bucketed models in legacy_models.py). Here are some of the key design differences for comparison:

|               | BucketModel | DynamicBot |
| ------------  | ----------    | -----------------------   |
| Embedding     | Used TensorFlow's ```EmbeddingWrapper```, which computes the embedding on a batch at each timestep. | Uses custom Embedder class to dynamically embed full batch-concatenated inputs of variable sequence length. |
| Encoding      | Employed the standard 'bucketed' model as described in TensorFlow sequence-to-sequence tutorial. Requires inputs to be padded to the same sequence length, for each bucket, which can result in unnecessarily large matrices of mainly zeros. | Combines the functionality of the new dynamic_rnn method in Tensorflow r1.0, wrapped inside a custom Encoder class. Input sequences are first fed to a custom batch_padded preprocessing utility (see utils/io_utils) that drastically reduces the occurrence of zero-padded sequences and allows for variable-length sequence batches. |
| Chatting      | Requires output to be assigned to a bucket, which constrains the raw output sequences to be constrained to pre-defined lengths. They then have to be truncated to remove padding. | Responses are generated naturally: once DynamicBot has read your input, it writes its response word by word until it signals that it's done speaking. No awkward post-processing required, and faster response times. |

One particular feature of DynamicBot worth mentioning is that the output generation and sampling process is _fully contained within the graph_ structure itself. This is in contrast with methods of outputting large arrays representing the logits (unnormalized log probabilities) and then sampling/argmax-ing over these. DynamicBot, however, directly returns its generated responses as a sequence of word-tokens.

## The Input Pipeline

In the past couple days, the way the model reads and interacts with data has been completely reimplemented. Before, a data generator fed padded numpy array batches from files to the model directly. It turns out that it is *substantially* faster encode the input information and preprocessing techniques in the graph structure itself. In the new implementation, we don't feed the model anything at all. Rather, it uses a sequence of queues to access the data from files in google's protobuf format, decode the files into tensor sequences, dynamically batch and pad the sequences, and then feed these batches to the embedding decoder. All within the graph structure. Furthermore, this data processing is coordinated by multiple threads in parallel. The result? You start the model, watch all cores of your CPU light up in a brief burst, and then it's 100% GPU training utilization (with helper CPU threads managing data in the background) for the remaining time (this is, of course, as reported on my system). 

Below are plots of training error for non-optimal hyperparameters on the ubuntu (purple), reddit (light blue), and cornell (orange) datasets. These plots correspond to just __five minutes__ of training time on each dataset. The relative performance is also expected; the current reddit data is our smallest and noisiest dataset (still working on the preprocessing) and cornell is about as high quality as one would ever need (grammatically correct movie dialogs). 

![](http://i.imgur.com/cM35tYJ.png)

Before the input pipeline, achieving these losses, even with finely tuned hyperparameters after running random search, would've taken around an hour or so. To be completely honest, I still have yet to determine why the training performance is this consistently better after only modifying the input structure. This is what I'll be analyzing for the next couple days. 


## Preliminary Testing

Now that the goals for DynamicBot have been met design-wise, I'm digging into the first big testing/debugging stage.

### Check 1: Ensure a large DynamicBot can overfit a small dataset.

Below is a plot related to one of the debugging strategies recommended in chapter 11 of *Deep Learning* by Goodfellow et al. The idea is that any sufficiently large model should be able to perfectly fit (well, overfit) a small training dataset. I wanted to make sure DynamicBot could overfit before I started implementing any regularizing techniques. It is a plot in TensorBoard of cross-entropy loss (y-axis) against global training steps (x-axis). The orange curve is the training loss, while the blue curve is the validation loss. TensorBoard has visually smoothed out the oscillations a bit. 

![Ensuring DynamicBot can overfit before optimizing any further](http://i.imgur.com/TLYvhEE.png)

This plot shows DynamicBot can achieve 0 loss for an extremely small dataset. Great, we can overfit. Now we can begin to explore regularization techniques.

### Check 2: Random & Grid Search Plots

I recently did a small random search and grid search over the following hyperparameters: learning rate, embed size, state size. The plots below show some of the findings. These are simply exploratory, I understand their limitations and I'm not drawing strong conclusions from them. They are meant to give a rough sense of the energy landscape in hyperparameter space. Oh and, plots make me happy. Enjoy. For all below, the y-axis is validation loss and the x-axis is global (training) step. The colors distinguish between model hyperparameters defined in the legends.

<img alt="state_size" src="http://i.imgur.com/w479tSo.png" width="400" align="left">
<img alt="embed_size" src="http://i.imgur.com/2Tj3vmA.png" width="400">
<br/>
<br/>


The only takeaway I saw from these two plots (after seeing the learning rate plots below) is that the __learning rate__, not the embed size, is overwhelmingly for responsible for any patterns here. It also looks like models with certain emed sizes (like 30) were underrepresented in the sampling, we see less points for them than others. The plots below illustrate the learning rate dependence.

<img alt="learning_subs" src="http://i.imgur.com/bD8MFrV.png" width="900">

**General conclusion: the learning rate influences the validation loss far more than state size or embed size.** This was basically known before making these plots, as it is a well known property of such networks (Ng). It was nice to verify this for myself.

