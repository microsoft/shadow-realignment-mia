#!/bin/bash

META_MODEL_SEED=1111

for R in {0..9}
do
    # MIAs using activation features from different layers.
	for TARGET_LAYER in fc2 fc2-ia fc1
	do
		python shadow_modeling_attack.py \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset_align_after_init  \
			--target_model_features=activations  \
			--target_model_layers=$TARGET_LAYER  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED
	        META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
    # MIAs using gradient features from different layers.
	for TARGET_LAYER in fc2 fc1 conv2 conv1
	do
		python shadow_modeling_attack.py \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset_align_after_init  \
			--target_model_features=gradients  \
			--target_model_layers=$TARGET_LAYER  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED
	        META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
	# MIAs using all features from different combinations of layers.
	for COMBINED_LAYERS in fc2 fc2-ia fc2-ia,fc1 fc2-ia,fc1,conv2,conv1
	do
		python shadow_modeling_attack.py \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset_align_after_init  \
			--target_model_features=activations,gradients  \
			--target_model_layers=$COMBINED_LAYERS  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED
		META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
done

