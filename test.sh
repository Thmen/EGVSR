#!/usr/bin/env bash

# This script is used to test pretrained models. More specific settings can
# be found and modified in a test.yml file under the experiment dir
#
# Usage:
#   bash test.sh <degradation> <model>
#
# Arguments:
#   degradation  - degradation type, e.g., BD or BI
#   model        - model name, e.g., EGVSR, TecoGAN, FRVSR, VESPCN, SOFVSR
#
# Examples:
#   bash test.sh BD EGVSR
#   bash test.sh BI TecoGAN
#
# The script expects the following directory structure:
#   ./experiments_<degradation>/<model>/001/test.yml

# basic settings
degradation=$1
model=$2
gpu_id=0
exp_id=001

if [ -z "$degradation" ] || [ -z "$model" ]; then
  echo "Usage: bash test.sh <degradation> <model>"
  echo ""
  echo "Arguments:"
  echo "  degradation  - degradation type (BD or BI)"
  echo "  model        - model name (EGVSR, TecoGAN, FRVSR, VESPCN, SOFVSR)"
  echo ""
  echo "Examples:"
  echo "  bash test.sh BD EGVSR"
  echo "  bash test.sh BI TecoGAN"
  exit 1
fi

exp_dir=./experiments_${degradation}/${model}/${exp_id}

if [ ! -d "$exp_dir" ]; then
  echo "Error: experiment directory not found: $exp_dir"
  exit 1
fi

if [ ! -f "$exp_dir/test.yml" ]; then
  echo "Error: test.yml not found in: $exp_dir"
  exit 1
fi

echo "Testing model: ${model} (degradation: ${degradation})"
echo "Experiment dir: ${exp_dir}"

# run
python ./codes/main.py \
  --exp_dir ${exp_dir} \
  --mode test \
  --model ${model} \
  --opt test.yml \
  --gpu_id ${gpu_id}
