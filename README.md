# Conversation Models in Tensorflow

This project is still very much evolving each day, but the core goals are:
* Recreate the results described in the paper, [A Neural Conversation Model](https://arxiv.org/pdf/1506.05869.pdf).
* Create a cleaner user interface for tinkering with such models and using multiple datasets. Although the release of TensorFlow 1.0 included great improvements in the API for sequence-to-sequence models, there are plenty of further improvements to be made. This project will explore ways to make constructing such models feel more intuitive/customizeable. The ideal result is a chatbot API with the readability of Keras, but with a degree of flexibility closer to TensorFlow. For example, the following code is all that is needed (after imports, etc.) to create and train one of the models on the Cornell movie dialogs:
```python
    # (Optional) Number of training samples used per gradient update.
    batch_size = 64
    # (Optional) Specify the max allowed number of words per sentence.
    max_seq_len = 500

    # All supported datasets inherit from a 'Dataset' ABC.
    dataset = Cornell(vocab_size=20000)

    # Example parameters for bot creation. More available.
    bot = DynamicBot(dataset, batch_size=batch_size, max_seq_len=max_seq_len)

    # All models have a "compile" function, inspired by Keras & with similar meaning.
    # See dynamic_models.py for supported parameters.
    bot.compile()

    # Get the desired data subset and reformat for training.
    encoder_inputs, decoder_inputs = data_utils.batch_concatenate(
        dataset.train_data, batch_size, max_seq_len
    )

    # Train an epoch on the data. CTRL-C at any time to safely stop training.
    # Model saved in "./out" by default. To modify, just pass in
    # 'save_dir=<your preferred directory>" as another argument in train.
    bot.train(encoder_sentences, decoder_sentences, steps_per_ckpt=FLAGS.steps_per_ckpt)
```



* Explore how [personalities of chatbots](https://arxiv.org/pdf/1603.06155.pdf) change when trained on different datasets, and methods for improving speaker consistency.
* Add support for "teacher mode": an interactive chat session where the user can tell the bot how well they're doing, and suggest better responses that the bot can learn from.

At present, the following have been (more or less) completed:

* Models:
    * (In progress) Rewriting conversation model with faster embedding technique and new TF support for dynamic unrolling. 
    * (Unsupported since transitioning to TF r1.0) Implement an attention-based embedding sequence-to-sequence model with the help of the tensorflow.contrib libraries.
    * (Unsupported since transitioning to TF r1.0) Implement a simpler embedding sequence-to-sequence from "scratch" (minimal use of contrib).
* Datasets:
    * **WMT'15** : English-to-French translation.
    * **Ubuntu Dialogue Corpus**: Reformatted as single-turn to single-response pairs.
    * **Cornell Movie-Dialogs**: Recently (March 5) incorporated [this preprocessed](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/cornell_movie_corpus) version of the Cornell corpus. I'll be processing and reformatting it further.


