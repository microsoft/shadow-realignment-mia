#!/bin/bash

#Train the proxy models using this command.
python shadow_modeling_attack.py --attacker_access=stolen_memories \
	--which_models=shadow   \
	--start_layer=-1

META_MODEL_SEED=99999

# Train an attack using the input activations (IA) of the last layer only, using no 
# re-alignment or top-down weight-based re-alignment.
for R in {0..9}
do
	python shadow_modeling_attack.py  --experiment=train_meta_model \
                --attacker_access=shadow_dataset \
                --target_model_features=activations \
                --target_model_layers=fc2-ia-only \
                --meta_model_target_exp=$R  \
                --meta_model_seed=$META_MODEL_SEED
	META_MODEL_SEED=$((META_MODEL_SEED+1))	
	python shadow_modeling_attack.py  --experiment=train_meta_model \
	       	--attacker_access=shadow_dataset \
	      	--target_model_features=activations \
	       	--target_model_layers=fc2-ia-only \
	       	--alignment=True \
	      	--alignment_method=top_down_weight_matching \
		--meta_model_target_exp=$R  \
		--meta_model_seed=$META_MODEL_SEED
	META_MODEL_SEED=$((META_MODEL_SEED+1))
done

# Run the Stolen Memories attack.
for R in {0..9}
do
	python shadow_modeling_attack.py --attacker_access=stolen_memories \
		--experiment=stolen_memories \
		--meta_model_target_exp=$R \
		--start_layer=-1
done
