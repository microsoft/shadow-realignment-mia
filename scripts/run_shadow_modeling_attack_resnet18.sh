#!/bin/bash

# Replace model config with vgg16.ini or vgg16-regularized_wd003_robust.ini

META_MODEL_SEED=0

GPU_ID=0

for R in {0..4}
do
	# Attack using activations.
	for LAYER in fc1 fc1-ia fc1-ia-only
	do
		for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/tiny-imagenet-200/resnet18.ini \
				--experiment=train_meta_model  \
				--attacker_access=$ATTACKER_ACCESS  \
				--target_model_features=activations  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED  \
				--num_target_models=5 \
				--num_shadow_models=2
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
	done
	python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
			--model_config=configs/tiny-imagenet-200/resnet18.ini \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset  \
			--target_model_features=activations  \
			--target_model_layers=fc1-ia-only  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED  \
			--alignment=True \
			--alignment_method=top_down_weight_matching \
			--num_shadow_models=2
	META_MODEL_SEED=$((META_MODEL_SEED+1))
	# Attack using gradients.
	for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
	do 
		python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
			--model_config=configs/tiny-imagenet-200/resnet18.ini \
			--experiment=train_meta_model  \
			--attacker_access=$ATTACKER_ACCESS  \
			--target_model_features=gradients  \
			--target_model_layers=fc1 \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED \
			--num_target_models=5 \
			--num_shadow_models=2
		META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
	python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
			--model_config=configs/tiny-imagenet-200/resnet18.ini \
			--experiment=train_meta_model  \
			--attacker_access=shadow_dataset  \
			--target_model_features=gradients  \
			--target_model_layers=fc1  \
			--meta_model_target_exp=$R  \
			--meta_model_seed=$META_MODEL_SEED \
			--alignment=True \
			--alignment_method=top_down_weight_matching \
			--num_shadow_models=2
	META_MODEL_SEED=$((META_MODEL_SEED+1))
	# Attack using activations and gradients.
	for LAYER in fc1-ia fc1
	do
		for ATTACKER_ACCESS in target_dataset shadow_dataset shadow_dataset_model_init
		do 
			python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/tiny-imagenet-200/resnet18.ini \
				--experiment=train_meta_model  \
				--attacker_access=$ATTACKER_ACCESS  \
				--target_model_features=activations,gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED \
				--num_target_models=5 \
				--num_shadow_models=2
			META_MODEL_SEED=$((META_MODEL_SEED+1))
		done
		python vgg_shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
				--model_config=configs/tiny-imagenet-200/resnet18.ini \
				--experiment=train_meta_model  \
				--attacker_access=shadow_dataset  \
				--target_model_features=activations,gradients  \
				--target_model_layers=$LAYER  \
				--meta_model_target_exp=$R  \
				--meta_model_seed=$META_MODEL_SEED \
				--alignment=True \
				--alignment_method=top_down_weight_matching \
				--num_shadow_models=2
		META_MODEL_SEED=$((META_MODEL_SEED+1))
	done
done
