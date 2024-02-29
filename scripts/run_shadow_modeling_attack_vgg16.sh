#!/bin/bash

# Replace model config with vgg16.ini or vgg16-regularized_wd003_robust.ini

META_MODEL_SEED=0

GPU_ID=0

for R in {0..9}
do
	# Attack using activations.
	for LAYER in fc3 fc3-ia fc2 fc1
	do
		for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=$ATTACKER_ACCESS  \
				--target_model_features=activations  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done
	
        # Attack using gradients.
	for LAYER in fc3 fc2 fc1
	do  
		for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
		do
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=$ATTACKER_ACCESS  \
				--target_model_features=gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done 
	# Attack using activations and gradients.
	for LAYER in fc3 fc3-ia fc3-ia,fc2
	do
		for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=$ATTACKER_ACCESS  \
				--target_model_features=activations,gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done
	# MIAs after applying re-alignment techniques to the models.
	# Attack using activations.
	for LAYER in fc2 fc1
	do
		for ALIGNMENT_METHOD in bottom_up_weight_matching bottom_up_activation_matching top_down_weight_matching
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=shadow_dataset  \
				--target_model_features=activations  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED  \
				--alignment=True \
				--alignment_method=$ALIGNMENT_METHOD
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done
	# Attack using gradients.
	for LAYER in fc3 fc2 fc1
	do  
		for ALIGNMENT_METHOD in bottom_up_weight_matching bottom_up_activation_matching top_down_weight_matching
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=shadow_dataset  \
				--target_model_features=gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED \
				--alignment=True \
				--alignment_method=$ALIGNMENT_METHOD
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done 
	# Attack using activations and gradients.
	for LAYER in fc3 fc3-ia fc3-ia,fc2
	do
		for ALIGNMENT_METHOD in bottom_up_weight_matching bottom_up_activation_matching top_down_weight_matching
		do
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/cifar10/vgg16.ini \
				--experiment=train_meta_model  \
				--attacker_access=shadow_dataset  \
				--target_model_features=activations,gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED \
				--alignment=True \
				--alignment_method=$ALIGNMENT_METHOD
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done
done
