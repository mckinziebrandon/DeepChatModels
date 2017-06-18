# Conversation Models in Tensorflow

Notes to visitors:
* I've just shut down the website indefinitely. I ran out of my credits on Google Cloud four days ago, and have since been billed 30+ dollars which isn't something I can sustain. To run locally, assuming you satisfy all requirements in webpage/requirements.txt, just run `python3 manage.py runserver`. If you're unfamiliar with running flask this way, see the docs for [Flask-Script](https://flask-script.readthedocs.io/en/latest/). Sorry for any inconvenience!
* Please post any feedbacks/bugs as an issue and I will respond within 24 hours. Speaking of headaches, the recent 1.2 release (changes [here](https://github.com/tensorflow/tensorflow/blob/master/RELEASE.md)) took quite a blow at the RNN libraries. I'm working on it, but I'd also happily accept a PR that accomodates the changes (while still being backwards-compatible). Currently, my branch `tf1.2-fixes-issue-6` makes the repo tf1.2 compatible, but breaks it for past versions.
* I haven't gotten around to providing scripts for downloading the datasets. Until then, I've uploaded most of the data [here on my MEGA account](https://mega.nz/#F!xrRTwSzY!by9K42n_I_oi5T_DKP-xTA). It is organized the same way I have it locally.
* Don't let the simple web bots fool you -- this project supports more advanced techniques than the single-layer encoder-decoder models on the website. To see the parameters that are immediately available/supported for tweaking, checkout chatbot/globals.py, which contains the default configuration dictionary. Any value that you don't specify will assume the default value from that file, which tend toward safe conversative simple values.
* Contributions are more than welcome. I do my best to follow PEP8 and I'd prefer contributions do the same.

## Table of Contents
* [Project Overview](#brief-overview-of-completed-work)
  * [Datasets](#datasets)
  * [Models](#models)
  * [Website](#website)
* [Model Components](#model-components)
  * [Input Pipeline](#the-input-pipeline)
* [Reference Material](#reference-material)

## Project Overview

As of May 9, 2017, the main packages of the project are as follows:
* __chatbot__: The conversation model classes, the structural components of the models (encoders, decoders, cells, etc.), and various operations for easy saving/loading/evaluation.
* __data__: The core Dataset class that handles all data formatting, file paths, and utilities for interacting with the data, as well as some preprocessing scripts and helper classes for cleaning data. The data itself (for space reasons) is not included in the repository. See the link to my MEGA account to download the data in the same format as on my local machine.
* __notebooks__: Jupyter notebooks showcasing data visualization examples, data preprocessing techniques, and conversation model exploration.
* __webpage__: Flask web application hosted on Google App Engine, where you can talk with a handful of chatbots and interact with plots. You can run it locally, after installing its requirements (mostly Flask packages), by running the following command within the webpage directory: `python3 manage.py runserver`

From a user/developer standpoint, this project offers a cleaner interface for tinkering with sequence-to-sequence models. The ideal result is a chatbot API with the readability of [Keras](https://keras.io/), but with a degree of flexibility closer to [TensorFlow](https://www.tensorflow.org/). 

On the 'client' side, playing with model parameters and running them is as easy as making a configuration (yaml) file, opening a python interpreter, and issuing a handful of commands. The following snippet, for example, is all that is needed to start training on the cornell dataset (after downloading it of course) with your configuration:
    
  ```python
    import data
    import chatbot
    from utils import io_utils
    # Load config dictionary with the flexible parse_config() function, 
    # which can handle various inputs for building your config dictionary.
    config = io_utils.parse_config(config_path='path_to/my_config.yml')
    dataset = getattr(data, config['dataset'])(config['dataset_params'])
    bot = getattr(chatbot, config['model'])(dataset, config)
    bot.train()
  ```
  
This is just one way to interface with the project. For example, the user can also pass in parameters via command-line args, which will be merged with any config files they specify as well (precedence given to command-line args if conflict). You can also pass in the location of a previously saved chatbot to resume training it or start a conversation. See `main.py` for more details.

### Datasets

* [Ubuntu Dialogue Corpus](https://arxiv.org/pdf/1506.08909.pdf): pre-processing approach can be seen in the ubuntu\_reformat.ipynb in the notebooks folder. The intended use for the dataset is response ranking for multi-turn dialogues, but I've taken the rather simple approach of extracting utterance-pairs and interpreting them as single-sentence to single-response, which correspond with inputs for the encoder and decoder, respectively, in the models.

* [Cornell Movie-Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html): I began with [this preprocessed](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus) version of the Cornell corpus, and made minor modifications to reduce noise.

* [Reddit comments](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/): Approx. 1.7 billion reddit comments. Currently working on preprocessing and reducing this massive dataset to suitable format for training conversation models. Will post processed dataset download links when complete!

### Models

* DynamicBot: uses a more object-oriented approach offered by custom classes in model_components.py. The result is faster online batch-concatenated embedding and a more natural approach to chatting. It makes use of the (fantastic) new python API in the TensorFlow 1.0 release, notably the dynamic_rnn. It also adheres to good variable scoping practice and common tensorflow conventions I've observed in the documentation and source code, which has nice side effects such as clean graph visualizations in TensorBoard.

* SimpleBot: Simplified bucketed model based on the more complicated 'ChatBot' model below. Although it is less flexible in customizing bucket partitions and uses a sparse softmax over the full vocabulary instead of sampling, it is far more transparent in its implementation. It makes minimal use of tf.contrib, as opposed to ChatBot, and more or less is implemented from "scratch," in the sense of primarily relying on the basic tensorflow methods. If you're new to TensorFlow, it may be useful to read through its implementation to get a feel for common conventions in tensorflow programming, as it was the result of me reading the source code of all methods in ChatBot and writing my own more compact interpretation.

* ChatBot: Extended version of the model described in [this TensorFlow tutorial](https://www.tensorflow.org/tutorials/seq2seq). Architecture characteristics: bucketed inputs, decoder uses an attention mechanism (see page 52 of my [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf), and inputs are embedded with the simple functions provided in the tf.contrib library. Also employs a sampled softmax loss function to allow for larger vocabulary sizes (page 54 of [notes](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf)). Additional comments: due to the nature of bucketed models, it takes much longer to create the model compared to others. The main bottleneck appears to be the size of the largest bucket and how the gradient ops are created based on the bucket sizes.

### Website

The webpage directory showcases a simple and space-efficient way for deploying your TensorFlow models in a Flask application. The models are 'frozen' -- all components not needed for chatting (e.g. optimizers) are removed and all remaining variables are converted to constants. When the user clicks on a model name, a REST API for that model is created. When the user enters a sentence into the form, an (AJAX) POST request is issued, where the response is the chatbot's response sentence. For more details on the REST API, see [views.py](https://github.com/mckinziebrandon/DeepChatModels/blob/master/webpage/deepchat/main/views.py).

The Flask application follows best practices, such as using blueprints for instantiating applications, different databases depending on the application environment (e.g. development or production), and more. 

## Model Components

Here I'll go into more detail on how the models are constructed and how they can be visualized. This section is a work in progress and not yet complete.

### The Input Pipeline

Instead of using the ```feed_dict``` argument to input data batches to the model, it is *substantially* faster encode the input information and preprocessing techniques in the graph structure itself. This means we don't feed the model anything at training time. Rather the model uses a sequence of queues to access the data from files in google's protobuf format, decode the files into tensor sequences, dynamically batch and pad the sequences, and then feed these batches to the embedding decoder. All within the graph structure. Furthermore, this data processing is coordinated by multiple threads in parallel. We can use tensorboard (and best practices for variable scoping) to visualize this type of pipeline at a high level.  

<img alt="input_pipeline" src="http://i.imgur.com/xrLqths.png" width="400" align="left">
<img alt="input_pipeline_expanded" src="http://i.imgur.com/xMWB7oL.png" width="400">
<br/>
<br/>

_(More descriptions coming soon!)_

## Reference Material

A lot of research has gone into these models, and I've been documenting my notes on the most "important" papers here in the last section of [my deep learning notes here](http://mckinziebrandon.me/assets/pdf/CondensedSummaries.pdf). The notes also include how I've tried translating the material from the papers into TensorFlow code. I'll be updating that as the ideas from more papers make their way into this project.

* Papers:
    * [Sequence to Sequence Learning with Neural Networks. Sutskever et al., 2014.](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
    * [On Using Very Large Target Vocabulary for Neural Machine Translation. Jean et al., 2014.](https://arxiv.org/pdf/1412.2007.pdf)
    * [Neural Machine Translation by Jointly Learning to Align and Translate. Bahdanau et al., 2016](https://arxiv.org/pdf/1409.0473.pdf)
    * [Effective Approaches to Attention-based Neural Machine Translation. Luong et al., 2015](https://arxiv.org/pdf/1508.04025.pdf)

* Online Resources:
    * [Metaflow blog](https://blog.metaflow.fr/): Incredibly helpful tensorflow (r1.0) tutorials.
    * [Flask Mega-Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world): For the webpage parts of the project.
    * [Code for "Massive Exploration of Neural Machine Translation Architectures"](https://github.com/google/seq2seq): Main inspiration for switching to yaml configs and pydoc.locate. Paper is great as well.
    * [Tensorflow r1.0 API](https://www.tensorflow.org/api_docs/): (Of course). The new python API guides are great. 
