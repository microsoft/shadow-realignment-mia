#!/bin/bash

CONFIG=$1
DATASET_SIZE=$2

python train_controlled_randomness.py   --model_config=$CONFIG   \
	--varying=seed_model   \
	--dataset_size=$DATASET_SIZE

python train_controlled_randomness.py   --model_config=$CONFIG   \
	--varying=seed_batching   \
	--dataset_size=$DATASET_SIZE

python train_controlled_randomness.py   --model_config=$CONFIG   \
	--varying=seed_dropout   \
	--dataset_size=$DATASET_SIZE

python train_controlled_randomness.py   --model_config=$CONFIG   \
	--varying=seed_all   \
	--dataset_size=$DATASET_SIZE


