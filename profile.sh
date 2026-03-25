#!/usr/bin/env bash

# This script is used to probe a network (generator), including the number of
# parameters, FLOPs, and the actual running speed.
#
# Usage:
#   bash profile.sh <degradation> <model> <lr_size>
#
# Arguments:
#   degradation  - degradation type, e.g., BD or BI
#   model        - model name, e.g., EGVSR, TecoGAN, FRVSR, VESPCN, SOFVSR
#   lr_size      - input size in format [color]x[height]x[width], e.g., 3x540x960
#
# Examples:
#   bash profile.sh BD EGVSR 3x270x480
#   bash profile.sh BI TecoGAN 3x540x960

# basic settings
degradation=$1
model=$2
lr_size=$3
exp_id=001
gpu_id=0

if [ -z "$degradation" ] || [ -z "$model" ] || [ -z "$lr_size" ]; then
  echo "Usage: bash profile.sh <degradation> <model> <lr_size>"
  echo ""
  echo "Arguments:"
  echo "  degradation  - degradation type (BD or BI)"
  echo "  model        - model name (EGVSR, TecoGAN, FRVSR, VESPCN, SOFVSR)"
  echo "  lr_size      - input size as [color]x[height]x[width] (e.g., 3x540x960)"
  echo ""
  echo "Examples:"
  echo "  bash profile.sh BD EGVSR 3x270x480"
  echo "  bash profile.sh BI TecoGAN 3x540x960"
  exit 1
fi

# run
python ./codes/main.py \
  --exp_dir ./experiments_${degradation}/${model}/${exp_id} \
  --mode profile \
  --model ${model} \
  --opt test.yml \
  --gpu_id ${gpu_id} \
  --lr_size ${lr_size} \
  --test_speed
