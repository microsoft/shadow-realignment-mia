#!/bin/bash

META_MODEL_SEED=505900

GPU_ID=0

# Run the attack against the CIFAR10 dataset.
for R in {0..9}
do
	# Attack using shadow models trained with a different initialization.
	python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
		--experiment=train_meta_model \
		--attacker_access=shadow_dataset  \
		--target_model_features=activations,gradients  \
		--target_model_layers=fc2-ia,fc1  \
		--set_based=True  \
		--meta_model_target_exp=$R   \
		--meta_model_seed=$META_MODEL_SEED 	
        META_MODEL_SEED=$((META_MODEL_SEED+1))
done

# Run the attack against the Texas100 dataset.
for R in {0..9}
do
        # Attack using shadow models trained with a different initialization.
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
                --experiment=train_meta_model \
                --model_config=configs/texas100/mlp_dropout.ini \
		--attacker_access=shadow_dataset  \
                --target_model_features=activations,gradients  \
                --target_model_layers=fc5-ia,fc4  \
                --set_based=True  \
		--num_set_based_features=103  \
                --meta_model_target_exp=$R   \
                --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
done

# Run the attack against the Purchase100 dataset.
for R in {0..9}
do
        # Attack using shadow models trained with a different initialization.
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
                --experiment=train_meta_model \
                --model_config=configs/purchase100/mlp_small_dropout.ini \
                --attacker_access=shadow_dataset  \
                --target_model_features=activations,gradients  \
                --target_model_layers=fc4-ia,fc3  \
                --set_based=True  \
                --num_set_based_features=103  \
                --meta_model_target_exp=$R   \
                --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
done

