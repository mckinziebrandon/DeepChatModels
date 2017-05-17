"""Place all default/global chatbot variables here."""

import tensorflow as tf

OPTIMIZERS = {
    'Adagrad':  tf.train.AdagradOptimizer,
    'Adam':     tf.train.AdamOptimizer,
    'SGD':      tf.train.GradientDescentOptimizer,
    'RMSProp':  tf.train.RMSPropOptimizer,
}

# All allowed and/or used default configuration values, period.
DEFAULT_FULL_CONFIG = {
    "model": "DynamicBot",
    "dataset": "Cornell",
    "model_params": {
        "base_cell": "GRUCell",
        "ckpt_dir": "out",  # Directory to store training checkpoints.
        "decode": False,
        "batch_size": 256,
        "dropout_prob": 0.2,  # Drop rate applied at encoder/decoders output.
        "decoder.class": "BasicDecoder",
        "encoder.class": "BasicEncoder",
        "embed_size": 128,
        "learning_rate": 0.002,
        "l1_reg": 1.0e-6,  # L1 regularization applied to word embeddings.
        "lr_decay": 0.98,
        "max_gradient": 5.0,
        "max_steps": int(1e6),  # Max number of training iterations.
        "num_layers": 1,  # Num layers for each of encoder, decoder.
        "num_samples": 512,  # IF sampled_loss is true, default sample size.
        "optimizer": "Adam",  # Options are those in OPTIMIZERS above.
        "reset_model": True,
        "sampled_loss": False,  # Whether to do sampled softmax.
        "state_size": 512,
        "steps_per_ckpt": 200,
        "temperature": 0.0,  # Response temp for chat sessions. (default argmax)
    },
    "dataset_params": {
        "data_dir": None,  # Require user to specify.
        "vocab_size": 40000,
        "max_seq_len": 10,  # Maximum length of sentence used to train bot.
        "optimize_params": True  # Reduce vocab size if exceeds num unique words
    },
}
