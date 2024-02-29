#!/bin/bash

META_MODEL_SEED=1042

GPU_ID=0

# Repeat the attack against 10 target models.
for R in {0..9}
do
    # MIAs using features extracted from the target model (scenario (S1) in the paper). 
    # First, we use activations from different layers.
    for LAYER in fc2 fc2-ia fc1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID  \
            --experiment=train_meta_model  \
            --attacker_access=target_dataset  \
            --target_model_features=activations  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Second, we use gradients from different layers.
    for LAYER in fc2 fc1 conv2 conv1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
            --experiment=train_meta_model  \
            --attacker_access=target_dataset  \
            --target_model_features=gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # MIAs using features extracted from shadow models trained with the same initialization as the R-th target model (scenario (S2) in the paper).
    # First, we use activations from different layers.
    for LAYER in fc2 fc2-ia fc1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset_model_init  \
            --target_model_features=activations  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Second, we use gradients from different layers.
    for LAYER in fc2 fc1 conv2 conv1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset_model_init  \
            --target_model_features=gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # MIAs using features extracted from shadow models trained with a different initialisation (classical adversary) from the R-th target model (scenario (S3) in the paper).
    # First, we use activations from different layers. 
    for LAYER in fc2 fc2-ia fc1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset  \
            --target_model_features=activations  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
        # We also run the attack with features extracted from re-aligned shadow models.
        for ALIGNMENT_METHOD in weight_sorting bottom_up_weight_matching  top_down_weight_matching bottom_up_activation_matching bottom_up_correlation_matching
        do
            python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
                --experiment=train_meta_model  \
                --attacker_access=shadow_dataset  \
                --target_model_features=activations  \
                --target_model_layers=$LAYER  \
                --alignment=True  \
                --alignment_method=$ALIGNMENT_METHOD  \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
            META_MODEL_SEED=$((META_MODEL_SEED+1))
        done
    done
    # Second, we use gradients from different layers. 
    for LAYER in fc2 fc1 conv2 conv1
    do
        python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset  \
            --target_model_features=gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
        # We also run the attack with features extracted from re-aligned shadow models.
        for ALIGNMENT_METHOD in weight_sorting bottom_up_weight_matching  top_down_weight_matching bottom_up_activation_matching bottom_up_correlation_matching
        do
            python shadow_modeling_attack.py  --gpu_id=$GPU_ID \
                --experiment=train_meta_model  \
                --attacker_access=shadow_dataset  \
                --target_model_features=gradients  \
                --target_model_layers=$LAYER  \
                --alignment=True  \
                --alignment_method=$ALIGNMENT_METHOD  \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
            META_MODEL_SEED=$((META_MODEL_SEED+1))
        done
    done
done
