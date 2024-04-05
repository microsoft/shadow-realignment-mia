# Investigating the Effect of Misalignment on Membership Privacy in the White-box Setting

## Description
Source code for reproducing the main results of the [paper](https://arxiv.org/abs/2306.05093) (to appear in PoPETs 2024). This includes modular code for training shadow models and membership inference attacks using different type of features, implementations of re-alignment techniques, and scripts for running the experiments. All the code is ours except for the code implementing ResNet, located in `src/resnet.py`, which is adapted from https://github.com/weiaicunzai/pytorch-cifar100.

## Basic Requirements

### Hardware Requirements
A Linux machine with one or more GPUs having Nvidia CUDA drivers >= 10.2 installed.

Estimated storage for all the datasets and experiments is 33G. 

### Software Requirements

#### Libraries

The machine should have `conda` installed. Instructions to install conda on Linux are provided [here](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

We also recommend installing and using `screen` so that (1) scripts can be left to run (some of them will take several days) and (2) multiple scripts can be run in parallel.

#### Datasets
We used six datasets in the paper: four small-scale datasets: CIFAR10, CIFAR100, Texas100, and Purchase100 and two large-scale datasets, CelebA, TinyImagenet-200, and are only used in secondary experiments. We provide instructions for downloading all of them, and flag as OPTIONAL the datasets that are used in secondary experiments.

- CIFAR10 and CIFAR100 will be downloaded automatically when attempting to run code (such as our scripts below) that trains shadow models on these datasets.
- Texas100 and Purchase100: Download them from https://www.comp.nus.edu.sg/~reza/files/datasets.html. Unzip their content into two folders (which you have to create), 'data/texas-100' and 'data/purchase-100'. Then run the following notebooks, `notebooks/texas.ipynb` and `notebooks/purchase.ipynb`, to process the datasets and save them to `.pickle` format (which allows for faster loading). The notebooks should be run _after_ having set up the environment (as described in the next section).

To run the notebooks, you should create a jupyter notebook instance on the server, from the current directory, e.g.:

```bash
jupyter notebook --no-browser --port=8888
```

then, if you are connecting via `ssh` to the server, run

```
ssh -N -L localhost:8889:localhost:8888 SERVER_ADDRESS
```

and finally open the browser and type in ```http://localhost:8889```.

- (OPTIONAL) CelebA: Download the dataset files from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (Baidu drive) and save them to a new directory called `data/celeba`, which should have the following contents:

```
identity_CelebA.txt
list_attr_celeba.txt
list_landmarks_align_celeba.txt
list_bbox_celeba.txt             
img_align_celeba.zip
list_eval_partition.txt
```

Unzip `img_align_celeba.zip` into the directory. This should create a new sub-directory  `data/celeba/img_align_celeba`. _Note:_ The unzipping will take a long time as there are many files.

- (OPTIONAL) TinyImageNet-200: Download the dataset from http://cs231n.stanford.edu/tiny-imagenet-200.zip into the `data` folder, then unzip it. This should create a new directory `data/tiny-imagenet-200`. _Note:_ The unzipping will take a long time as there are many files.

### Estimated Time and Storage Consumption
It will take up to 4 weeks to run all the experiments using one GPU, however if the machine has multiple GPUs it is possible to run the scripts in parallel on different GPUs. In this case, the running time will go down proportionally.

The expected storage requirement for the datasets and experiments is 33G: we provide in `storage_requirement_breakdown.md` a breakdown for the `data` and `experiments` directories. 

## Environment

### Set up the environment

Clone this repository.

```
git clone git@github.com:microsoft/shadow-realignment-mia.git
```

Run the commands below to create a conda environment and install all required packages, except for `torch` (which requires special treatment):

```
conda create --name wb-mia python=3.10.4
conda activate wb-mia
pip install numpy==1.22.4 scikit-learn==1.3.2 scipy==1.11.4 configargparse tqdm hungarian_algorithm matplotlib jupyter ipykernel pandas
python -m ipykernel install --user --name wb-mia --display-name "wb-mia"
```

Next, attempt to install the same version of `torch` that we used in our experiments:

```
pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu102
```

Attempt to run the following command:

```
python train_controlled_randomness.py --model_config=configs/cifar10/tests/cnn-large-test-controlled-randomness.ini  --varying=seed_model  --num_experiments=3
```

If this returns an error, such as `RuntimeError: CUDA error: no kernel image is available for execution on the device`, it means that this older `torch` version is incompatible with the GPU you're using. We ran into this issue when attempting to install this library version on an NVIDIA GeForce RTX 4050 GPU, although we did not run into the issue on NVIDIA TITAN Xp, NVIDIA Tesla V100 and V100S GPUs (probably because they are older). If you encounter a CUDA-related  error, uninstall `torch` and install a newer version:

```
pip uninstall torch==1.11.0+cu102 torchvision==0.12.0+cu102 torchaudio==0.11.0 
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Re-run the test. If you encounter an error, it means that the GPU is incompatible with the `torch` version. In this case, you will have to append the argument `--use_gpu=False` in all of the scripts below, to run everything on the CPU (not recommended as it will take much longer).

### Testing the Environment
Run the test scripts below to make sure everything is working as expected. The scripts should complete without error and you may ignore their output.

Activate the environment (if not already activated).
```
conda activate wb-mia
```

Test 1: Train two models with controlled randomness, such that they are trained on the same dataset, same batch ordering, same dropout sampling but with a different weight initialisation. This should be very quick as the training set size is set to 5000 samples and the training runs only for one epoch. 

The test is successful if the scripts complete without error. The saved models will be saved in  ```experiments/cifar10/controlled_randomness/cnn-large-test-controlled-randomness/dsize-5000/```.

```
python train_controlled_randomness.py --model_config=configs/cifar10/tests/cnn-large-test-controlled-randomness.ini  --varying=seed_model  --num_experiments=3
```

Test 2: Train a target model, then extract features from an internal layer to run an MIA. The target model is a toy model trained for 1 epochs on 5000 samples. As we train the MIA classifier for 1 epoch (only for testing puurposes), the MIA will achieve random performance. 

The test is successful if both scripts below complete without error.

```
python shadow_modeling_attack.py --model_config=configs/cifar10/tests/cnn-large-test-mia.ini  --which_models=target --attacker_access=target_dataset   --num_target_models=1
```

The target model performance is printed at the end of training and should be close to the following (can be different due to difference in hardware and `torch` version leading to different randomness):

```
Train acc: 25.0%, val acc: 26.8%, test acc: 24.7%
```

The saved model will be saved in `experiments/cifar10/attack/cnn-large-test-mia/target_models/`. 

```
python shadow_modeling_attack.py --model_config=configs/cifar10/tests/cnn-large-test-mia.ini  --attacker_access=target_dataset   --num_target_models=1  --experiment=train_meta_model   --target_model_features=activations  --target_model_layers=fc1  --meta_model_max_num_epochs=1
```

The saved membership classifier with attack results will be saved in `experiments/cifar10/attack/cnn-large-test-mia/attack_results/aa-target_dataset/activations/fc1`

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Causes of shadow model misalignment
Our first result (Sec. 4) is that the main cause for the misalignment of shadow models trained by the classical adversary (described in Sec. 2.2) is their different weight initialisation. 
We disentangle the impact on misalignment of the different sources of ML randomness as well as the impact of training shadow models on a different dataset.
We show that when the adversary uses a different weight initialisation for shadow models than the target model's, the former end up misaligned with the latter.
Conversely, an adversary having knowledge of the target model's initialisation is able to train shadow models which are internally more similar to the target model.

Table 1 demonstrates this result on a standard CNN architecture trained on the CIFAR10 dataset, where we measure misalignment via the weight misalignment score (described in Sec. 4.1). 
Table 13 in the Appendix shows similar results using two different metrics for measuring misalignment: the activation misalignment score and correlation between activations (described in Appendix A.1).
We provide in Experiment 1 instructions for reproducing Table 1, Table 13 (Appendix), Figure 1 and Figures 3-5 (Appendix).

We further replicate this finding in Appendix A.1 on:
1. The CIFAR100 dataset using the same CNN architecture (Table 9)
2. The Purchase100 dataset using an MLP architecture (Table 10)
3. The CIFAR10 dataset using different training set sizes (Table 11 and Figure 6)
4. The CIFAR10 dataset using different training hyperparameters (Table 12)
5. The CelebA dataset using same training distribution between shadow and target models or a different training distribution between shadow and target models (Table 14 and Figure 2).

We provide in Experiment 4-8 instructions on how to run these experiments, which we flag as OPTIONAL.

#### Main Result 2: Re-alignment techniques can reduce misalignment
Our second result (Sec. 5) is that re-alignment techniques are able to reduce the misalignment between the target model and the shadow models. We implement and evaluate the effectiveness of several re-alignment techniques in reducing misalignment. Misalignment is measured using the weight misalignment score (Table 2), the activation misalignment score (Table 15A) and the correlation between activations (Table 15B).

We provide in Experiment 2 instructions for reproducing Table 2 and Table 15 (Appendix).

#### Main Result 3: In-depth study of white-box MIA performance across features types and layers
Our third result (Sec. 6) is an in-depth study of white-box MIAs using features of different types (output activations, gradients, and input activations of the last layer), extracted from different layers, as well as combined features extracted from multiple layers. 

These results take a long time to run (up to 2-3 weeks), since we run 10 repetitions of the attack in each setting, which amounts to training hundreds of shadow models and membership classifiers for every dataset. Thus, we recommend running Experiment 3 to obtain results on the standard CNN architecture trained on CIFAR10, which will give Tables 4, line 1 of Table 6, Table 7, and Figure 7 (Appendix). In Experiments 9-12 we provide instructions for the other datasets, marking them as OPTIONAL.

### Experiments

#### Experiment 1: Causes of shadow model misalignment
The experiment consists in training several models with controlled randomness and training data, then measuring the misalignment between one of the models (referred to as the target model) and the other models (referred to as the shadow models) using three different metrics. More specifically, computing each line of Table 1 requires training one target model and several shadow models using the same conditions, except for one or more factors we control for. For instance, the results in line 2 of Table 1 (Different weight initialisation) are computed from 1 target model and 5 shadow models trained in the same way: same dataset, batch ordering and dropout sampling, except for the use of a different weight initialisation.

The script below will train all the models necessary to generate Table 1, Table 13 (Appendix), Figures 1 and 3-5. Estimated running time is up to 1-2 days, the models will be saved to `experiments/cifar10/controlled_randomness/cnn-large/dsize-12500`, and the estimated disk space is 120M.

```
bash scripts/table1_train_controlled_randomness.sh configs/cifar10/cnn-large.ini
```

To generate the results of Table 1, we need to compute the weight misalignment metrics. This should be done by re-running *one-by-one* each of the commands of 
`scripts/table1_train_controlled_randomness.sh` after adding to it `--experiment=compute_metrics`. For instance, run

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_model   --dataset_size=12500  --experiment=compute_metrics
```

to compute the weight misalignment metric (Table 1) but also the two other metrics reported in Table 13 for models trained with a different weight initialisation from the target model.


The results should be similar to the following (possibly different due to differences in hardware/randomness):

```
Weight misalignment score (Random permutation) for layer 1: 11.81 (0.51); layer 2: 16.58 (0.15); layer 3: 31.87 (0.06); layer 4: 12.80 (0.66)
Weight misalignment score w.r.t. seed_model for layer 1: 12.09 (0.34); layer 2: 16.24 (0.20); layer 3: 30.46 (0.45); layer 4: 12.54 (0.14)

Activation misalignment score (Random Permutation) for layer 1: 68.59 (3.10); layer 2: 72.70 (0.65); layer 3: 28.60 (0.05); layer 4: 1.02 (0.12)
Activation misalignment score w.r.t. seed_model for layer 1: 68.31 (1.96); layer 2: 70.99 (1.46); layer 3: 27.34 (0.53); layer 4: 0.32 (0.01)

Correlation between activations (Random permutation) for  layer 1: 0.15 (0.07); layer 2: 0.04 (0.02); layer 3: 0.01 (0.00); layer 4: 0.04 (0.05)
Correlation between activations w.r.t. seed_model for layer 1: 0.13 (0.03); layer 2: 0.03 (0.01); layer 3: 0.01 (0.00); layer 4: 0.28 (0.07)
```

From this output we can extract line 1 of Table 1 (Random permutation), line 2 of Table 1 (Different weight initialisation), lines 1-2 of Table 13A and lines 1-2 of Table 13B.

To obtain line 3-9 of Tables 1, 13A and 13B, run every command listed in `scripts/table1_train_controlled_randomness.sh` after adding to it `--experiment=compute_metrics`, like above.

To generate Figure 1 and Figures 3-5, run the following notebooks: ```notebooks/figure1.ipynb``` and ```notebooks/figures3-5.ipynb```.

#### Experiment 2: Re-alignment techniques can reduce misalignment
Using the models trained in the previous experiment, we now compute the misalignment metrics between the target model and re-aligned shadow models. We apply several re-alignment techniques to shadow models trained by the classical adversary (i.e., which were trained on a disjoint dataset and using different weight initialisation, batch ordering and dropout sampling w.r.t. the target model). Our results,  showing that re-alignment techniques can reduce the misalignment, are reported in Table 2 (weight misalignment metric), Table 15A (activation misalignment metric), and Table 15B (correlation between activations). We describe below how to generate the results.

Line (A0) of Tables 2, 15A and 15B is the same as Line 9 of Tables 1, 13A, and 13B respectively.

Line (A1) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=12500  --experiment=compute_metrics  --alignment=True    --alignment_method=weight_sorting
```

Line (A2) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint_align_after_init   --dataset_size=12500
```
to train shadow models re-aligned to the target model after the initialisation, then by running

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint_align_after_init   --dataset_size=12500  --experiment=compute_metrics
```
to compute the misalignment metrics.

Line (A3) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=12500  --experiment=compute_metrics  --alignment=True    --alignment_method=bottom_up_weight_matching
```

Line (A4) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=12500  --experiment=compute_metrics  --alignment=True    --alignment_method=top_down_weight_matching
```

Line (A5) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=12500  --experiment=compute_metrics  --alignment=True    --alignment_method=bottom_up_activation_matching
```

Line (A6) of Tables 2, 15A and 15B can be obtained by running:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=seed_all_dataset_disjoint   --dataset_size=12500  --experiment=compute_metrics  --alignment=True    --alignment_method=bottom_up_correlation_matching
```

_Note:_ For re-alignment techniques (A5) and (A6) we set `--alignment_method` to `bottom_up_activation_matching` and `bottom_up_correlation_matching`, respectively, while these methods are referred to in the paper as "Activation-based re-alignment" and "Correlation-based re-alignment". In the paper, we explain in Sec. 5 that for these re-alignment techniques, the order in which the optimal permutations are computed doesn't matter, i.e., top-down and bottom-up are equivalent. In the code however, we performed them--without loss of generalisation--in bottom-up order.

#### Experiment 3: In-depth study of white-box MIA performance across features types and layers
This experiment consists in (1) training the target and shadow models and (2) evaluating MIAs using different types of features. The features can be extracted from the target model (scenario (S1) in the paper), from shadow models trained using the same weight initialisation as the target model (S2), or from shadow models trained using a different weight initialisation (S3). Furthermore, the features can be of different types and extracted from different layers.

The results of this experiment will be saved in `experiments/cifar10/attack/cnn-large/shadow_models`, `experiments/cifar10/attack/cnn-large/target_models`, and `experiments/cifar10/attack/cnn-large/attack_results` of size 863M, 28M, and 119M, respectively.

Run the commands below to train:

1. Target models (10 repetitions)
```
python shadow_modeling_attack.py --model_config=configs/cifar10/cnn-large.ini --which_model=target --attacker_access=target_dataset  --num_target_models=10
```

2. 10 shadow models using different weight initialisation (the same models will be reused across the 10 repetitions).
```
python shadow_modeling_attack.py --model_config=configs/cifar10/cnn-large.ini --which_model=shadow --attacker_access=shadow_dataset  --num_target_models=10  --num_shadow_models=10
```

3. 10 shadow models using the same weight initisalisation as the target model, for every target model (100 models in total).
```
python shadow_modeling_attack.py --model_config=configs/cifar10/cnn-large.ini --which_model=shadow --attacker_access=shadow_dataset_model_init  --num_target_models=10  --num_shadow_models=10
```

Run the scripts below to train the membership classifier using different features, layers, and combinations thereof:

1. MIAs using features from one layer only.
```
bash scripts/run_shadow_modeling_attack_cifar10_layer_by_layer.sh
```

2. MIAs combining features from multiple layers.
```
bash scripts/run_shadow_modeling_attack_cifar10_combined.sh
```

_Note:_ If you plan to run these scripts in parallel, modify the $GPU_ID variable in the scripts to make sure they are running on different GPUs.

To aggregate the results, run the notebook below:

```notebooks/results_shadow_modeling_attack_cifar10.ipynb```

We provide a mapping between layer names used in the bash commands and the way they are referred to in the paper:

- Black-box MIA using output activations (OA), reported in Table 6 (fourth column) and Table 7 (first column): `--target_model_layers=fc2`, `--target_model_features=activations`.
- White-box MIA using gradients (G) of the last layer, reported in Table 4: `--target_model_layers=fc2`, `--target_model_features=gradients`.
- White-box MIA using OA of the second to last layer, reported in Table 4: `--target_model_layers=fc1`, `--target_model_features=activations`.
- White-box MIA using G of the  second to last layer, reported in Table 4: `--target_model_layers=fc1`, `--target_model_features=gradients`.
- White-box MIA using G of the third to last layer, reported in Table 4: `--target_model_layers=conv2`, `--target_model_features=gradients`.
- White-box MIA using G of the fourth to last layer, reported in Table 4: `--target_model_layers=conv1`, `--target_model_features=gradients`.
- White-box MIA using OA + input activations (IA) + G of the last layer, reported in Table 6 (first column) and Table 7 (second column):  `--target_model_layers=fc2-ia`, `--target_model_features=activations,gradients`.
- White-box MIA using OA + G of the last layer, reported in Table 6:  `--target_model_layers=fc2`, `--target_model_features=activations,gradients`.
- White-box MIA using OA + IA of the last layer, reported in Table 6:  `--target_model_layers=fc2-ia`, `--target_model_features=activations`.
- White-box MIA using all features in the last two layers, reported in Table 7:  `--target_model_layers=fc2-ia,fc1`, `--target_model_features=activations,gradients`.
- White-box MIA using all features in all four layers, reported in Table 7:  `--target_model_layers=fc2-ia,fc1,conv2,conv1`, `--target_model_features=activations,gradients`.

3. (OPTIONAL) Tables 4 and 7 also report results for the "re-alignment after initialisation" technique (S9). To save time and resources, we do not recommend running this experiment, as it performs poorly (as acknowledged in the paper) and it will require training 100 new shadow models (that are re-aligned to the target after initialisation). For completeness, we provide instructions below if you wish to run this experiment:

```
python shadow_modeling_attack.py --model_config=configs/cifar10/cnn-large.ini --which_model=shadow --attacker_access=shadow_dataset_align_after_init  --num_target_models=10  --num_shadow_models=10
```

```
bash scripts/run_shadow_modeling_attack_cifar10_align_after_init.sh
```

As before, the results can be aggregated in ```notebooks/results_shadow_modeling_attack_cifar10.ipynb```.

#### (OPTIONAL) Experiment 4: Causes of misalignment on CIFAR100
Table 9 reports the weight misalignment metric (WMS) computed on CNN models trained on CIFAR100. To reproduce these results, train the models using the command below, then compute the WMS using similar commands as in Experiment 1 (making sure to use the correct config file, given below):

```
bash scripts/table1_train_controlled_randomness.sh configs/cifar100/cnn-large.ini
```

The results of this experiment will be saved in `experiments/cifar100/controlled_randomness/cnn-large/dsize-12500` of size 116M.

####  (OPTIONAL) Experiment 5: Causes of misalignment on Purchase100
Table 10 reports the weight misalignment metric (WMS) computed on MLP models trained on Purchase100. To reproduce these results, train the models using the command below, then compute the WMS using similar commands as in Experiment 1 (making sure to use the correct config file, given below):

```
bash scripts/table1_train_controlled_randomness.sh configs/purchase100/controlled_randomness/mlp_small_dropout.ini
```

The results of this experiment will be saved in `experiments/purchase100/controlled_randomness/generic-mlp-dropout_600,512,256,128,100/dsize-20000` of size 92M.

####  (OPTIONAL) Experiment 6: Causes of misalignment on CIFAR10 using different training set sizes (Table 11 and Figure 6)
Table 11 reports the weight misalignment metric (WMS) computed on CNN models trained on varying number of samples of CIFAR10: 12500 (taken from Experiment 1), 25000, and 50000. We provide instructions for obtaining the last two:


```
bash scripts/train_controlled_randomness.sh configs/cifar10/cnn-large.sh 25000
```

Setting the second argument to 0 amounts to using the entire dataset (here, 50000 records).
```
bash scripts/train_controlled_randomness.sh configs/cifar10/cnn-large.sh 0
```

Once the models are trained, you can use commands similar to the ones provided in Experiment 1 to compute the WMS. In this case, it suffices to append `--experiment=compute_metrics` to every command in `./scripts/train_controlled_randomness.sh`.

Figure 6 shows a visualisation on first layer activation maps for models trained on varying number of samples (1250 to 50000). To reproduce this figure, run:

```
python train_controlled_randomness.py   --model_config=configs/cifar10/cnn-large.ini   --varying=dataset_sizes   --dataset_size=0
```

then run the notebook ```notebooks/figure6.ipynb```.

####  (OPTIONAL) Experiment 7: Causes of misalignment on CIFAR10 using different training hyperparameters (Table 12)

The first three rows of Table 12 are computed from models trained with a learnin rate of 0.01 and early stopping patience of 5. They are the same as rows 11-13 of Table 11.

The next three rows are computed from models trained with a learning rate of 0.001 and early stopping patience of 5.
```
bash scripts/train_controlled_randomness.sh configs/cifar10/cnn-large-lr-0.001_e-5.sh 0
```

The next three rows are computed from models trained with a learning rate of 0.001 and early stopping patience of 10.
```
bash scripts/train_controlled_randomness.sh configs/cifar10/cnn-large-lr-0.001_e-10.sh 0
```

####  (OPTIONAL) Experiment 8: Misalignment in models trained on different distributions from the CelebA dataset (Table 14 and Figure 2)

```
bash scripts/celeba_train_controlled_randomness.sh
```

To compute the weight misalignment scores reported in Table 14, run the same commands as in the script after adding `--experiment=compute_metrics` to them.

Run the `notebooks/figure2.ipynb` to plot Figure 2. 

#### (OPTIONAL) Experiment 9: White-box MIAs against VGG16 trained on CIFAR10 (Tables 5 and 19)
Train the target and shadow models using the commands below:

```
python shadow_modeling_attack.py --model_config=configs/cifar10/vgg16.ini --which_model=target --attacker_access=target_dataset  --num_target_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/cifar10/vgg16.ini --which_model=shadow --attacker_access=shadow_dataset  --num_target_models=10  --num_shadow_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/cifar10/vgg16.ini --which_model=shadow --attacker_access=shadow_dataset_model_init  --num_target_models=10  --num_shadow_models=10
```

Run the MIAs using the script below:

```
bash scripts/run_shadow_modeling_attack_vgg16.sh
```

Aggregate the MIA results using `notebooks/results_shadow_modeling_attack_vgg16.ipynb`

#### (OPTIONAL) Experiment 10: White-box MIAs against MLP models trained on Texas100 (Table 8)
Train the target and shadow models using the commands below:

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_dropout.ini --which_model=target --attacker_access=target_dataset  --num_target_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_dropout.ini --which_model=shadow --attacker_access=shadow_dataset  --num_target_models=10  --num_shadow_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_dropout.ini --which_model=shadow --attacker_access=shadow_dataset_model_init  --num_target_models=10  --num_shadow_models=10
```

Run the MIAs using the script below:

```
bash scripts/run_shadow_modeling_attack_texas100.sh
```

Aggregate the MIA results using `notebooks/results_shadow_modeling_attack_texas100.ipynb`

#### (OPTIONAL) Experiment 11: White-box MIAs against MLP models trained on Purchase100 (Table 20)
Train the target and shadow models using the commands below:

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_small_dropout.ini --which_model=target --attacker_access=target_dataset  --num_target_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_small_dropout.ini --which_model=shadow --attacker_access=shadow_dataset  --num_target_models=10  --num_shadow_models=10
```

```
python shadow_modeling_attack.py --model_config=configs/texas100/mlp_small_dropout.ini --which_model=shadow --attacker_access=shadow_dataset_model_init  --num_target_models=10  --num_shadow_models=10
```

Run the MIAs using the script below:

```
bash scripts/run_shadow_modeling_attack_purchase100.sh
```

Aggregate the MIA results using `notebooks/results_shadow_modeling_attack_texas100.ipynb`

#### (OPTIONAL) Experiment 12: White-box MIAs against Resnet18 models trained on Tiny-Imagenet-200 (Table 21)
Train the target and shadow models using the commands below:

```
python vgg_shadow_modeling_attack.py  --model_config=configs/tiny-imagenet-200/resnet18.ini   --attacker_access=target_dataset --which_model=target --num_target_models=5
```

```
python vgg_shadow_modeling_attack.py  --model_config=configs/tiny-imagenet-200/resnet18.ini   --attacker_access=shadow_dataset --which_model=shadow --num_target_models=5  --num_shadow_models=2
```

```
python vgg_shadow_modeling_attack.py  --model_config=configs/tiny-imagenet-200/resnet18.ini   --attacker_access=shadow_dataset_model_init --which_model=shadow --num_target_models=5  --num_shadow_models=2

```

Run the MIAs using the script below:

```
bash scripts/run_shadow_modeling_attack_resnet18.sh
```

Aggregate the MIA results using `notebooks/results_shadow_modeling_attack_resnet18.ipynb`

#### (OPTIONAL) Experiment 13: Other baselines (Tables 16, 17, and 18) and model utility (Table 3)

Table 16: Instructions can be found in `scripts/run_stolen_memories.sh`. 

Table 17: Instructions can be found in `scripts/run_shadow_modeling_attack_set_based.sh`

Table 18: Instructions can be found in `scripts/run_lira_attack.sh`. The results can be aggregated in `notebooks/results_lira_attack.sh`.

Model utility (Table 3): Run `notebooks/model_utility.ipynb`.

## Limitations
The artifact does not include the experiments for training models with differential privacy (DP, Appendix A.9), because the library we used for DP training (opacus) is incompatible with our version of CUDA and pytorch; changing the pytorch version enables DP training but leads to a segmentation error when running other scripts. We were not able to find a combination of versions for CUDA, pytorch, and opacus that would allow to run all the scripts, error-free, on our hardware, so we opted to release an environment that works for all the other experiments. 

We however provide the model config file for VGG16 architecture trained with regularisation defenses, `configs/cifar10/vgg16-regularized_wd003_robust.ini`. You can train VGG16 models using commands similar to Experiment 9 to check that MIAs achieve close to random performance against this model.

## Notes on Reusability
We hope that future work will extend the re-alignment techniques (`src/align.py`) to other networks and symmetries along the lines described in our paper.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
