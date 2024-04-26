We provide below a breakdown of the estimated running time per experiment on a Tesla V100 GPU.

**Experiment 1:** Training of 42 models * ~603s/model ~= 4h

**Experiment 2:** Total time ~= 40min
1. Computing the metrics: a couple of seconds for each command.
2. Training of 4 additional models (with re-alignment after initialisation) * ~603s/model ~= 40min

**Experiment 3:** Total time ~= 4 days
1. Training of 120 models * 13min/model ~= 1 day 2h
2. Script 1 for running MIAs ~= 2 days
3. Script 2 for running MIAs ~= 2 days
4. Training of 100 additional models (with re-alignment after initialisation) * 13min/model = 21h40min

**Experiment 4:** Training of 42 models * 7min/model ~= 5h

**Experiment 5:** Training of 42 models * 4min/model ~= 3h

**Experiment 6:** Total time ~ 1 day
1. Training of 24 models * 1100s/model ~= 7h20min
2. Training of 24 models * 2234s/model ~= 15h
3. Training of 6 models on varying dataset size ~= 1h15min




