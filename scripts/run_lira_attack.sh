#!/bin/bash

for R in {0..9}
do
       for ATTACKER_ACCESS in shadow_dataset shadow_dataset_model_init
       do
	       python shadow_modeling_attack.py --attacker_access=$ATTACKER_ACCESS --experiment=run_lira_attack --meta_model_target_exp=$R

	       python shadow_modeling_attack.py --attacker_access=$ATTACKER_ACCESS --experiment=run_lira_attack --meta_model_target_exp=$R  --model_config=configs/texas100/mlp_dropout.ini

	       python shadow_modeling_attack.py --attacker_access=$ATTACKER_ACCESS --experiment=run_lira_attack --meta_model_target_exp=$R  --model_config=configs/purchase100/mlp_small_dropout.ini
       done 
done
