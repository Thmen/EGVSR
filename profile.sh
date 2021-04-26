#!/usr/bin/env bash

# This script is used to probe a network (generator), including the number of
# parameters, FLOPs, and the actual running speed.


# basic settings
degradation=$1
model=$2
exp_id=001
gpu_id=0
# specify the size of input data in the format of [color]x[height]x[weight]
lr_size=$3


# run
python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode profile \
  --model ${model} \
  --opt test.yml \
  --gpu_id ${gpu_id} \
  --lr_size ${lr_size} \
  --test_speed
