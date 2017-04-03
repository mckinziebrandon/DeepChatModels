#!/bin/bash
# Simple wrapper script that suppresses unnecessary info logging from tensorflow.
# WARNING: DO NOT RUN THIS IF YOU ARE PASSING TF FLAGS AS COMMAND-LINE ARGS.
# -- They will not be parsed correctly by bash.

python3 $* 3>&1 1>&2 2>&3 3>&- | grep -v ^I\ | grep -v ^pciBusID | grep -v ^major: | grep -v ^name: |grep -v ^Total\ memory:|grep -v ^Free\ memory:
