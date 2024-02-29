#!/bin/bash

CONFIG=$1
GPU_ID=0

# Train the models used in line 2 of table 1 (Different weight initlisation).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_model   --dataset_size=12500

# Train the models used in line 3 of table 1 (Different batch ordering).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_batching   --dataset_size=12500

# Train the models used in line 4 of table 1 (Different dropout sampling).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_dropout   --dataset_size=12500

# Train the models used in line 5 of table 1 (Overlapping datasets).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=dataset_overlapping   --dataset_size=12500

# Train the models used in line 6 of table 1 (Disjoint datasets).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=dataset_disjoint   --dataset_size=12500

# Train the models used in line 7 of table 1 (Different batch ordering, dropout sampling, and disjoint dataset).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_batching_dropout_dataset_disjoint   --dataset_size=12500

# Train the models used in line 8 of table 1 (Different weight initisalisation, batch ordering, and dropout sampling).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_all   --dataset_size=12500

# Train the models used in line 9 of table 1 (Different weight initisalisation, batch ordering, dropout sampling, and disjoint dataset).
CUDA_VISIBLE_DEVICES=$GPU_ID python train_controlled_randomness.py   --model_config=$CONFIG   --varying=seed_all_dataset_disjoint   --dataset_size=12500
