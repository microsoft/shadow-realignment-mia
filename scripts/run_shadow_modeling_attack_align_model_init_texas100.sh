#!/bin/bash

META_MODEL_SEED=104256


# Activation + gradient-based attacks.
for ALIGNMENT_METHOD in top_down_weight_matching bottom_up_weight_matching
do
	for R in {0..9}
	do
		# Attack using re-aligned shadow models trained with the same 
		# initialization.
		python shadow_modeling_attack.py  \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset_model_init  \
			--model_config=configs/texas100/mlp_dropout.ini  \
			--target_model_features=activations,gradients  \
			--target_model_layers=fc5-ia  \
			--alignment=True \
			--alignment_method=$ALIGNMENT_METHOD  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED 
		META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
done

