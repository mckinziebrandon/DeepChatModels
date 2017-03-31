#!/bin/bash


python3 main.py \
    --config configs/test_config.yml \
    --model_params "
    decode: True
    reset_model: False
    "


