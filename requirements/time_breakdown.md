The expected running time for all the experiments is 18 days 2h using only one GPU. 

By running the scripts in parallel on multiple GPUs, the running time can be reduced proportionally.

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

**Experiment 7:** Total time ~ 1 day 16h
1. Training of 24 models * 2550s/model ~= 17h
2. Training of 24 models * 3360s/model ~= 23h
   
**Experiment 8:** Training of 16 models * 45min/model ~= 12h

**Experiment 9:** Total time ~= 4 days 2h
1. Training of 120 models * 25min/model ~= 2days 2h
2. Running the MIAs ~= 2 days

**Experiment 10:** Total time ~= 1 day 14h
1. Training of 120 models * 3min/model ~= 6h
2. Running the MIAs ~= 1day 8h

**Experiment 11:** Total time ~= 2 days 5h
1. Training of 120 models * 5 min/model ~= 10h
2. Running the MIAs ~= 1day 19h

**Experiment 12:** Total time ~= 1 day 9h
1. Training of 30 models * 19min/model ~= 9h30min
2. Running the MIAs ~= 23h

**Experiment 13:** Total time ~= 1day 3h
1. Training of 100 proxy models ~= 21h
2. Running the stolen memories attack ~= 1h
3. Running the set-based classifier attack ~= 5h
4. Running the LiRA attack ~ 30mins
