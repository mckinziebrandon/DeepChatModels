#!/bin/bash

# Amazing helper script to filter out all the annoying INFO log messages every time tensorflow starts. 
# Author: github user Fenugreek. Source: https://github.com/tensorflow/tensorflow/issues/566
python3 $* 3>&1 1>&2 2>&3 3>&- | grep -v ^I\ | grep -v ^pciBusID | grep -v ^major: | grep -v ^name: |grep -v ^Total\ memory:|grep -v ^Free\ memory:
