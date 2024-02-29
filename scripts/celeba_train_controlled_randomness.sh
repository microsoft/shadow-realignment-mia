#!/bin/bash

# (Adversary 1) Here, model #1 is the target model trained on a subset of old faces. 
# Model #2 is a model trained using  different randomness and disjoint subset of old faces.
python train_controlled_randomness.py   --model_config=configs/celeba-old/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=20000  --num_experiments=2

# (Adversary 2) Here, models #2-#6 are trained using a different seed and random subsets of faces (predominantly young) which are disjoint from the target model's training dataset.
python train_controlled_randomness.py   --model_config=configs/celeba/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=20000  --num_experiments=6

# Ablation.
# Here, model #2 is trained using same randomness as the target model but on a disjoint subset of old faces.
python train_controlled_randomness.py   --model_config=configs/celeba-old/cnn-large.ini   --varying=dataset_disjoint   --dataset_size=20000  --num_experiments=2

# Here, models #2-#6 are trained using the same randomness as the target model, but on random subsets of faces (predominantly young), which are mutually disjoint.
python train_controlled_randomness.py   --model_config=configs/celeba/cnn-large.ini   --varying=dataset_disjoint   --dataset_size=20000  --num_experiments=6




