# Conversation Models in Tensorflow

Note to visitors: data preprocessing work is still in the active stages and we haven't (yet) provided scripts for downloading the datasets. Updates will be posted here when this is completed.

__Table of Contents__
* [Brief Overview of Completed Work](#brief-overview-of-completed-work)
* [Faster Embedding, Encoding, and Chatting](#faster-embedding-encoding-and-chatting)
* [The Input Pipeline](#the-input-pipeline)
* [Preliminary Testing](#preliminary-testing)
    * [Overfitting](#check-1-ensure-a-large-dynamicbot-can-overfit-a-small-dataset)

This project is still very much evolving each day, but the core goals are:
* Create a cleaner user interface for tinkering with sequence-to-sequence models. This project will explore ways to make constructing such models feel more intuitive/customizable. The ideal result is a chatbot API with the readability of [Keras](https://keras.io/), but with a degree of flexibility closer to [TensorFlow](https://www.tensorflow.org/). 
  + Initially, the general API was as shown below, with named parameters being the primary way of tweaking model values. 
  ```python
    # All datasets implement a Dataset interface, found in data/_dataset.py
    dataset = Cornell(vocab_size=FLAGS.vocab_size)

    # Create chat model of choice. (Only a subset of available parameters shown). 
    print("Creating DynamicBot.")
    bot = DynamicBot(dataset,
                     ckpt_dir=FLAGS.ckpt_dir,
                     batch_size=FLAGS.batch_size,
                     state_size=FLAGS.state_size,
                     embed_size=FLAGS.embed_size)

    # Don't forget to compile! Name inspired by similar Keras method.
    print("Compiling DynamicBot.")
    bot.compile(max_gradient=FLAGS.max_gradient, reset=FLAGS.reset_model)

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in FLAGS.ckpt_dir if specified, else "./out"
    print("Training bot. CTRL-C to stop training.")
    bot.train(dataset.train_data, dataset.valid_data,
              nb_epoch=FLAGS.nb_epoch,
              steps_per_ckpt=FLAGS.steps_per_ckpt)
  ```
  + However, as the models began to grow it became clear that named parameters weren't the best option, especially considering that the typical user only interacts with a subset of them. The current solution, which allows for even more customization but with far less boilerplate is to load configuration (yaml) files. In addition, the user can also pass in parameters via command-line args, which will be merged with any config files they specify as well (precedence given to command-line args if conflict). The code for main.py, after building the configuration dictionary, is primarily:
    
  ```python
    print("Setting up %s dataset." % config['dataset'])
    dataset = locate(config['dataset'])(config['dataset_params'])
    print("Creating", config['model'], ". . . ")
    bot = locate(config['model'])(dataset, config)
  ```

which also makes use of `pydoc.locate`, a common practice I've seen for similar projects. See `main.py` for more details.

* Explore how [personalities of chatbots](https://arxiv.org/pdf/1603.06155.pdf) change when trained on different datasets, and methods for improving speaker consistency.
* Implement and improve "teacher mode": an interactive chat session where the user can tell the bot how well they're doing, and suggest better responses that the bot can learn from.

## Brief Overview of Completed Work


__Encoder/Decoder-Based Models__:
* DynamicBot: uses a more object-oriented approach offered by custom classes in model_components.py. The result is faster online batch-concatenated embedding and a more natural approach to chatting. It makes use of the (fantastic) new python API in the TensorFlow 1.0 release, notably the dynamic_rnn. It also adheres to good variable scoping practice and common tensorflow conventions I've observed in the documentation and source code, which has nice side effects such as clean graph visualizations in TensorBoard.

* SimpleBot: Simplified bucketed model based on the more complicated 'ChatBot' model below. Although it is less flexible in customizing bucket partitions and uses a sparse softmax over the full vocabulary instead of sampling, it is far more transparent in its implementation. It makes minimal use of tf.contrib, as opposed to ChatBot, and more or less is implemented from "scratch," in the sense of primarily relying on the basic tensorflow methods. If you're new to TensorFlow, it may be useful to read through its implementation to get a feel for common conventions in tensorflow programming, as it was the result of me reading the source code of all methods in ChatBot and writing my own more compact interpretation.

* ChatBot: Extended version of the model described in [this TensorFlow tutorial](https://www.tensorflow.org/tutorials/seq2seq). Architecture characteristics: bucketed inputs, decoder uses an attention mechanism (see page 52 of my [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf), and inputs are embedded with the simple functions provided in the tf.contrib library. Also employs a sampled softmax loss function to allow for larger vocabulary sizes (page 54 of [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf)). Additional comments: due to the nature of bucketed models, it takes much longer to create the model compared to others. The main bottleneck appears to be the size of the largest bucket and how the gradient ops are created based on the bucket sizes.


__Datasets__:
* [WMT'15](http://www.statmt.org/wmt15/translation-task.html): 22M sentences examples of english-to-french translation.

* [Ubuntu Dialogue Corpus](https://arxiv.org/pdf/1506.08909.pdf): pre-processing approach can be seen in the ubuntu\_reformat.ipynb in the notebooks folder. The intended use for the dataset is response ranking for multi-turn dialogues, but I've taken the rather simple approach of extracting utterance-pairs and interpreting them as single-sentence to single-response, which correspond with inputs for the encoder and decoder, respectively, in the models.

* [Cornell Movie-Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html): I began with [this preprocessed](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus) version of the Cornell corpus, and made minor modifications to reduce noise.

* [Reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/): Approx. 1.7 billion reddit comments. Currently working on preprocessing and reducing this massive dataset to suitable format for training conversation models. Will post processed dataset download links when complete!

__[Ongoing] Reference Material__: A lot of research has gone into these models, and I've been documenting my notes on the most "important" papers here in the last section of [my deep learning notes here](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf). I'll be updating that as the ideas from more papers make their way into this project. 

* Papers:
    * [Sequence to Sequence Learning with Neural Networks. Sutskever et al., 2014.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
    * [On Using Very Large Target Vocabulary for Neural Machine Translation. Jean et al., 2014.](https://arxiv.org/pdf/1412.2007.pdf)
    * [Neural Machine Translation by Jointly Learning to Align and Translate. Bahdanau et al., 2016](https://arxiv.org/pdf/1409.0473.pdf)
    * [Effective Approaches to Attention-based Neural Machine Translation. Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)

* Online Resources:
    * [Metaflow blog](https://blog.metaflow.fr/): Incredibly helpful tensorflow (r1.0) tutorials.
    * [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world): For the webpage parts of the project.
    * [Code for "Massive Exploration of Neural Machine Translation Architectures"](Massive Exploration of Neural Machine Translation Architectures): Main inspiration for switching to yaml configs and pydoc.locate. Paper is great as well.
    * [Tensorflow r1.0 API](https://www.tensorflow.org/api_docs/): (Of course). The new python API guides are great. 


## Faster Embedding, Encoding, and Chatting

The newest model, ```DynamicBot```, is substantially faster than the previous models (bucketed models in legacy_models.py). Here are some of the key design differences for comparison:

|               | BucketModel | DynamicBot |
| ------------  | ----------    | -----------------------   |
| Embedding     | Used TensorFlow's ```EmbeddingWrapper```, which computes the embedding on a batch at each timestep. | Uses custom Embedder class to dynamically embed full batch-concatenated inputs of variable sequence length. |
| Encoding      | Employed the standard 'bucketed' model as described in TensorFlow sequence-to-sequence tutorial. Requires inputs to be padded to the same sequence length, for each bucket, which can result in unnecessarily large matrices of mainly zeros. | Combines the functionality of the new dynamic_rnn method in Tensorflow r1.0, wrapped inside a custom Encoder class. Input sequences are first fed to a custom batch_padded preprocessing utility (see utils/io_utils) that drastically reduces the occurrence of zero-padded sequences and allows for variable-length sequence batches. |
| Chatting      | Requires output to be assigned to a bucket, which constrains the raw output sequences to be constrained to pre-defined lengths. They then have to be truncated to remove padding. | Responses are generated naturally: once DynamicBot has read your input, it writes its response word by word until it signals that it's done speaking. No awkward post-processing required, and faster response times. |

One particular feature of DynamicBot worth mentioning is that the output generation and sampling process is _fully contained within the graph_ structure itself. This is in contrast with methods of outputting large arrays representing the logits (unnormalized log probabilities) and then sampling/argmax-ing over these. DynamicBot, however, directly returns its generated responses as a sequence of word-tokens.

## The Input Pipeline

Instead of using the ```feed_dict``` argument to input data batches to the model, it is *substantially* faster encode the input information and preprocessing techniques in the graph structure itself. This means we don't feed the model anything at training time. Rather the model uses a sequence of queues to access the data from files in google's protobuf format, decode the files into tensor sequences, dynamically batch and pad the sequences, and then feed these batches to the embedding decoder. All within the graph structure. Furthermore, this data processing is coordinated by multiple threads in parallel. We can use tensorboard (and best practices for variable scoping) to visualize this type of pipeline at a high level.  

<img alt="input_pipeline" src="http://i.imgur.com/xrLqths.png" width="400" align="left">
<img alt="input_pipeline_expanded" src="http://i.imgur.com/xMWB7oL.png" width="400">
<br/>
<br/>

(Descriptions coming soon)

## Preliminary Testing

Now that the goals for DynamicBot have been met design-wise, I'm digging into the first big testing/debugging stage.

### Check 1: Ensure a large DynamicBot can overfit a small dataset.

Below is a plot related to one of the debugging strategies recommended in chapter 11 of *Deep Learning* by Goodfellow et al. The idea is that any sufficiently large model should be able to perfectly fit (well, overfit) a small training dataset. I wanted to make sure DynamicBot could overfit before I started implementing any regularizing techniques. It is a plot in TensorBoard of cross-entropy loss (y-axis) against global training steps (x-axis). The orange curve is the training loss, while the blue curve is the validation loss. TensorBoard has visually smoothed out the oscillations a bit. 

![Ensuring DynamicBot can overfit before optimizing any further](http://i.imgur.com/TLYvhEE.png)

This plot shows DynamicBot can achieve 0 loss for an extremely small dataset. Great, we can overfit. Now we can begin to explore regularization techniques.
