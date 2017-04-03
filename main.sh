#!/bin/bash
# Simple wrapper script that suppresses unnecessary info logging from tensorflow.
# Same usage as main.py.

./main.py $* 3>&1 1>&2 2>&3 3>&- | grep -v ^I\ | grep -v ^pciBusID | grep -v ^major: | grep -v ^name: |grep -v ^Total\ memory:|grep -v ^Free\ memory:
