#!/bin/bash

for lr in 3e-5 
do
  for train_batch_size in 4
  do
      outdir="/mnt/disks/sdb/dayoon_ko/output/nqpara-try"
      python train.py \
        --output_dir "$outdir" \
        --learning_rate "$lr" \
        --num_epochs 1 \
        --train_batch_size "$train_batch_size" \
        -d "dayoon/HDS-QA" \
        --shuffle True
  done
done


