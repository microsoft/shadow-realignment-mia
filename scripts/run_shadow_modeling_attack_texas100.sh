#!/bin/bash

META_MODEL_SEED=42

MODEL_CONFIG=configs/texas100/mlp_dropout.ini

# Activation-based attacks.
for R in {0..9}
do
    for LAYER in fc5 fc5-ia
    do
        # Attack using the target model.
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=target_dataset  \
            --target_model_features=activations  \
            --target_model_layers=$LAYER  \
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
            --target_model_layers=fc5  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
    # Attack using shadow models trained with a different initialization.
    python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --model_config=$MODEL_CONFIG  \
            --attacker_access=shadow_dataset  \
            --target_model_features=activations  \
            --target_model_layers=fc5  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
    META_MODEL_SEED=$((META_MODEL_SEED+1))
done

# Activation + gradient-based attacks.
for R in {0..9}
do
    # Attack using the target model.
    for LAYER in fc5-ia fc5
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=target_dataset  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Attack using shadow models trained with the same initialization.
    for LAYER in fc5-ia fc5
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset_model_init  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
    done
    # Attack using shadow models trained with a different initialization.
    for LAYER in fc5-ia fc5
    do
        python shadow_modeling_attack.py  \
            --experiment=train_meta_model  \
            --attacker_access=shadow_dataset  \
            --model_config=$MODEL_CONFIG  \
            --target_model_features=activations,gradients  \
            --target_model_layers=$LAYER  \
            --meta_model_target_exp=$R  \
            --meta_model_seed=$META_MODEL_SEED
        META_MODEL_SEED=$((META_MODEL_SEED+1))
        for ALIGNMENT_METHOD in weight_sorting bottom_up_weight_matching top_down_weight_matching top_down_activation_matching bottom_up_correlation_matching
        do
            python shadow_modeling_attack.py  \
                --experiment=train_meta_model  \
                --attacker_access=shadow_dataset  \
                --model_config=$MODEL_CONFIG  \
                --target_model_features=activations,gradients  \
                --target_model_layers=$LAYER  \
                --alignment=True  \
                --alignment_method=$ALIGNMENT_METHOD  \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
            META_MODEL_SEED=$((META_MODEL_SEED+1))
        done
    done
done

