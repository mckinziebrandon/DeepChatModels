#!/bin/bash

n=$1
cp train_from.txt train_from_full.txt
head -n -"$n" "train_from_full.txt" >> "train_from.txt"
tail -n "$n" "train_from_full.txt" >> "valid_from.txt"

cp train_to.txt train_to_full.txt
head -n -"$n" "train_to_full.txt" >> "train_to.txt"
tail -n "$n" "train_to_full.txt" >> "valid_to.txt"

rm *_full.txt
wc -l *
