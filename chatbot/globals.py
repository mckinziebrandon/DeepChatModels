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
        "attention_size": None,
        "base_cell": "GRUCell",
        "ckpt_dir": "out",
        "decode": False,
        "batch_size": 256,
        "dropout_prob": 0.2,
        "decoder.class": "BasicDecoder",
        "encoder.class": "BasicEncoder",
        "embed_size": 128,
        "learning_rate": 0.002,
        "l1_reg": 1.0e-6,
        "lr_decay": 0.98,
        "max_gradient": 5.0,
        "max_steps": int(1e6),
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
        "data_dir": None,  # Require user to specify.
        "vocab_size": 40000,
        "max_seq_len": 10,
        "optimize_params": True
    },
}
