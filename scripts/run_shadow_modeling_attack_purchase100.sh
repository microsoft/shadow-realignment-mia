#!/bin/bash

META_MODEL_SEED=42

MODEL_CONFIG=configs/purchase100/mlp_small_dropout.ini

# Black-box attack.
for R in {0..9}
do
    # Attacks using activations.
    for TARGET_MODEL_LAYERS in fc4 fc4-ia
    do
        # Attack using the target model.
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=target_dataset  \
            --target_model_features=activations  \
            --target_model_layers=$TARGET_MODEL_LAYERS  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Attack using shadow models trained with the same initialization.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=shadow_dataset_model_init  \
            --target_model_features=activations  \
            --target_model_layers=fc4  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
    # Attack using shadow models trained with a different initialization.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=shadow_dataset  \
            --target_model_features=activations  \
            --target_model_layers=fc4  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))

    # Attacks using gradients.
    # Attack using the target model.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=target_dataset  \
            --target_model_features=gradients  \
            --target_model_layers=fc4  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
    # Attack using shadow models trained with the same initialization.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=shadow_dataset_model_init  \
            --target_model_features=gradients  \
            --target_model_layers=fc4  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
    # Attack using shadow models trained with a different initialization.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=shadow_dataset  \
            --target_model_features=gradients  \
            --target_model_layers=fc4  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
    for ALIGNMENT_METHOD in weight_sorting bottom_up_weight_matching top_down_activation_matching top_down_weight_matching bottom_up_correlation_matching
    do
        python shadow_modeling_attack.py  \
                --experiment=train_meta_model  \
                --attacker_access=shadow_dataset  \
                --model_config=$MODEL_CONFIG  \
                --target_model_features=gradients  \
                --target_model_layers=fc4  \
                --alignment=True  \
                --alignment_method=$ALIGNMENT_METHOD  \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done

    # Attacks using activations and gradients.
    # Attack using the target model.
    for TARGET_MODEL_LAYERS in fc4 fc4-ia
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=target_dataset  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$TARGET_MODEL_LAYERS  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Attack using shadow models trained with the same initialization.
    for TARGET_MODEL_LAYERS in fc4 fc4-ia
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset_model_init  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$TARGET_MODEL_LAYERS  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Attack using shadow models trained with a different initialization.
    for TARGET_MODEL_LAYERS in fc4 fc4-ia
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$TARGET_MODEL_LAYERS  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
        for ALIGNMENT_METHOD in weight_sorting bottom_up_weight_matching top_down_activation_matching top_down_weight_matching bottom_up_correlation_matching
        do
            python shadow_modeling_attack.py  \
                --experiment=train_meta_model  \
                --attacker_access=shadow_dataset  \
                --model_config=$MODEL_CONFIG  \
                --target_model_features=activations,gradients  \
                --target_model_layers=$TARGET_MODEL_LAYERS  \
                --alignment=True  \
                --alignment_method=$ALIGNMENT_METHOD  \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
            META_MODEL_SEED=$((META_MODEL_SEED+1))
        done
    done
done

