#!/usr/bin/env bash

# This script is used to train a model. More specific setttings can
# be found and modified in a train.yml file under the experiment dir


# basic settings
degradation=$1
model=$2
exp_id=001
gpu_id=0


# retain training or train from scratch
start_iter=0
if [[ ${start_iter} -gt 0 ]]; then
	suffix=_iter${start_iter}
else
	suffix=''
fi


exp_dir=./experiments_${degradation}/${model}/${exp_id}
# check
if [ -d "$exp_dir/train" ]; then
  	echo "Train folder already exists: $exp_dir/train"
    echo "Please delete the train folder, or setup a new experiment"
    echo "Exiting ..."
  	exit 1
fi


# backup codes
mkdir -p ${exp_dir}/train
cp -r ./codes ${exp_dir}/train/codes_backup${suffix}


# run
python ./codes/main.py \
  --exp_dir ${exp_dir} \
  --mode train \
  --model ${model} \
  --opt train${suffix}.yml \
  --gpu_id ${gpu_id} \
  > ${exp_dir}/train/train${suffix}.log  2>&1 &
