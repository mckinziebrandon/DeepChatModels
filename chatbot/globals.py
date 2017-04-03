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
    "model": "chatbot.DynamicBot",
    "dataset": "data.Cornell",
    "model_params": {
        "ckpt_dir": "out",
        "decode": False,
        "batch_size": 256,
        "dropout_prob": 0.2,
        "decoder.class": "chatbot.components.decoders.BasicDecoder",
        "encoder.class": "chatbot.components.encoders.BasicEncoder",
        "embed_size": 64,
        "learning_rate": 0.002,
        "l1_reg": 1e-7,
        "lr_decay": 0.98,
        "max_gradient": 5.0,
        "num_layers": 3,
        "num_samples": 512,
        "optimizer": "Adam",
        "reset_model": True,
        "sampled_loss": False,
        "state_size": 512,
        "steps_per_ckpt": 200,
        "temperature": 0.0,
    },
    "dataset_params": {
        "data_dir": None,
        "vocab_size": 40000,
        "max_seq_len": 10
    },
}
