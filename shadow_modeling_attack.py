import configargparse
import math
import numpy as np
import os
import pickle
import scipy.stats as st
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.checkers import check_disjoint
from src.dataset import load_dataset, get_num_classes
from src.models import compute_average_model, get_architectures, init_model
from src.align import (GreedyMatching,
    HungarianAlgorithmMatching,
    WeightSortingBasedAlignment,
    BottomUpWeightMatchingBasedAlignment,
    TopDownWeightMatchingBasedAlignment,
    BottomUpActivationMatchingBasedAlignment,
    TopDownActivationMatchingBasedAlignment,
    BottomUpCorrelationMatchingBasedAlignment)
from src.train_shadow_model import train_shadow_model
from src.train_meta_model import train_meta_model
from src.train_proxy_model import train_proxy_model


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgumentParser(
        description='Train ML models by controlling for the various sources of randomness.')

    # Seed.
    parser.add_argument('--seed', type=int, default=0)

    # GPU id.
    parser.add_argument('--use_gpu', type=str2bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    # Attack scenario: one of `target_dataset`, `shadow_dataset_model_init`, 
    # `shadow_dataset`.
    parser.add_argument('--attacker_access', type=str, default='target_dataset')
    parser.add_argument('--num_target_models', type=int, default=10)
    parser.add_argument('--num_shadow_models', type=int, default=10)
    parser.add_argument('--num_val_records_shadow_model', type=int, default=2000)
    parser.add_argument('--which_models', type=str, default='target',
        help='Which models to train: target or shadow?')
    parser.add_argument('--alignment', type=str2bool, default=False)
    # Should be one of: "bottom_up_weight_matching", "weight_sorting", 
    # "bottom_up_activation_matching", "top_down_weight_matching".
    parser.add_argument('--alignment_method', type=str, 
        default='bottom_up_weight_matching')
    # Should be "greedy" or "hungarian_algorithm".
    parser.add_argument('--matching_method', type=str, 
            default='hungarian_algorithm')
    parser.add_argument('--num_records', type=int, default=500, \
        help='How many records to use for activation-based matching.')

    # Set this to a value larger than 0 if you want to artificially restrain the training
    # dataset size of the meta model, in the `target_dataset` attacker access scenario.
    parser.add_argument('--num_meta_model_train_records', type=int, default=0)
    parser.add_argument('--num_meta_model_val_records', type=int, default=2000)
    parser.add_argument('--num_meta_model_test_records', type=int, default=5000)
    parser.add_argument('--shadow_model_val_exp', type=int, default=0)
    
    # General parameters (seed, dataset, experiment).
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataset_size', type=int, default=0)
    parser.add_argument('--small', type=str2bool, default=False, 
        help='For the Purchase100 dataset, whether to use the small (balanced) version.')
    parser.add_argument('--transform', type=str, default='normalize')
    # Directory where the results are saved.
    parser.add_argument('--save_dir', type=str, default='experiments')
    # Should be one of: `train_models`, `train_meta_model`, or 
    # `stolen_memories`.
    parser.add_argument('--experiment', type=str, default='train_models')
    
    # Shadow model training parameters.
    parser.add_argument('--model_config', is_config_file=True, 
            default='configs/cifar10/cnn-large.ini')
    parser.add_argument('--architecture', type=str, default='cnn-large')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=9e-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_patience', type=int, default=5)
    parser.add_argument('--select_best_model', type=str2bool, default=True)
    parser.add_argument('--limit_overfitting', type=str2bool, default=False)

    # Stolen Memories
    parser.add_argument('--start_layer', type=int, default=-2,
        help='Layer to use for splitting model in Stolen Memories attack.')
    
    # Meta model training parameters.
    parser.add_argument('--meta_model_target_exp', type=int, default=0, 
        help='Which repetition of the attack to run, ie which target model to attack.')
    parser.add_argument('--meta_model_seed', type=int, default=42)
    parser.add_argument('--meta_model_batch_size', type=int, default=64)
    parser.add_argument('--meta_model_optimizer', type=str, default='adam')
    parser.add_argument('--meta_model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--meta_model_min_learning_rate', type=float, default=1e-4)
    parser.add_argument('--meta_model_momentum', type=float, default=9e-1)
    parser.add_argument('--meta_model_weight_decay', type=float, default=0)
    parser.add_argument('--meta_model_max_num_epochs', type=int, default=100)
    parser.add_argument('--meta_model_num_epochs_patience', type=int, default=1)
    parser.add_argument('--target_model_layers', type=str, default='fc2', 
        help='From which layer to extract features.')
    parser.add_argument('--target_model_features', type=str, 
            default='activations',
        help='Which features to use in the attack.')
    parser.add_argument('--meta_model_kernel_size', type=int, default=100, 
        help='Size of the kernel used in the CNN encoder for the gradients.')
    parser.add_argument('--meta_model_encoder_sizes', type=str, 
            default='128,64', help='Sizes of the encoder operating on the activations/gradient embeddings.')
    parser.add_argument('--set_based', type=str2bool, default=False)
    parser.add_argument('--num_set_based_features', type=int, default=13)

    parser.add_argument('--print_every', type=int, default=10)

    return parser.parse_args()


def init_parameters(args):
    dataset = load_dataset(args.dataset, args.transform, dataset_size=args.dataset_size, seed=args.seed, small=args.small)
    if dataset['val'] is not None: # True for CIFAR10 and CIFAR100.
        dataset = ConcatDataset( (dataset['train_and_test'], dataset['val']) )
    else:
        dataset =  dataset['train_and_test']
    num_records = len(dataset)
    print(f'Total number of records {num_records}')

    print('Initializing the global parameters of the experiment.')

    shadow_idxs, target_idxs = train_test_split(np.arange(num_records), test_size=0.5, random_state=args.seed)
    shadow_train_test_idxs = shadow_idxs[:-args.num_val_records_shadow_model]
    shadow_val_idxs = shadow_idxs[-args.num_val_records_shadow_model:]
    target_train_test_idxs = target_idxs[:-args.num_val_records_shadow_model]
    target_val_idxs = target_idxs[-args.num_val_records_shadow_model:]
    check_disjoint( (shadow_train_test_idxs, shadow_val_idxs, target_train_test_idxs, target_val_idxs) )

    idxs = {'shadow_train_test_idxs': shadow_train_test_idxs,
        'shadow_val_idxs': shadow_val_idxs,
        'target_train_test_idxs': target_train_test_idxs,
        'target_val_idxs': target_val_idxs}

    print(f'Length of splits: shadow train+test={len(shadow_train_test_idxs)}, shadow val={len(shadow_val_idxs)},',
        f'target train+test={len(target_train_test_idxs)}, target val={len(target_val_idxs)}')

    # Initialize the seeds.
    # Note: by using np.random.choice we ensure consistency even as we 
    # increase `--num_target_models` or `--num_shadow_models`.
    np.random.seed(args.seed)
    # The target models are trained and tested on the same dataset across the
    # experiments.
    seeds_target_splits = [int(np.random.choice(10**8))]*args.num_target_models
    #seeds_target_splits = np.random.choice(10**8, args.num_target_models, 
    #        replace=False)
    np.random.seed(args.seed+1)
    seeds_shadow_splits = np.random.choice(10**8, args.num_shadow_models, 
            replace=False)
    np.random.seed(args.seed+2)
    seeds_target_models = np.random.choice(10**8, 
            size=(args.num_target_models, 3), replace=False).transpose()
    np.random.seed(args.seed+3)
    seeds_shadow_models = np.random.choice(10**8, 
            size=(args.num_shadow_models, 3), replace=False).transpose()
    # TODO: Enforce that the seeds are distinct (no need for now as the seeds generated with default 
    # parameters pass the test.).
    check_disjoint( (seeds_shadow_models.reshape(-1).tolist(), 
        seeds_target_models.reshape(-1).tolist()) )
    seeds = {'target_splits': seeds_target_splits,
        'shadow_splits': seeds_shadow_splits,
        'target_models': seeds_target_models,
        'shadow_models': seeds_shadow_models}
    return dataset, idxs, seeds


def train_models(args):
    dataset, idxs, seeds = init_parameters(args)
    
    device = torch.device(f'cuda:{args.gpu_id}' 
            if torch.cuda.is_available() and args.use_gpu else 'cpu')

    save_dir = os.path.join(args.save_dir, args.dataset, 'attack', 
            args.architecture)

    for target_exp in range(args.num_target_models):
        if args.which_models == 'target':
            assert args.attacker_access == 'target_dataset', \
                    f'ERROR: Invalid --attacker_access={args.target_dataset}.'
            val = Subset(dataset, idxs['target_val_idxs'])
            target_save_dir = os.path.join(save_dir, 'target_models')
            print(f'Saving directory for the target model {target_exp+1}/{args.num_target_models}', target_save_dir)
            train_idxs, test_idxs = train_test_split(
                    idxs['target_train_test_idxs'], test_size=0.5, 
                    random_state=seeds['target_splits'][target_exp])
            #print(train_idxs[:5], test_idxs[:5])
            train, test = Subset(dataset, train_idxs), \
                    Subset(dataset, test_idxs)
            model_seeds = ( seeds['target_models'][0, target_exp], # Vary the model initialization.
                seeds['target_models'][1, target_exp], # Vary the batch sampling seed.
                seeds['target_models'][2, target_exp] # Vary the dropout seed
            )
            # Mapping the seeds to integers as required by torch.Generator.
            model_seeds = tuple([int(ms) for ms in model_seeds])
            # The indexes of train and test records will be saved to disk 
            # together with the model.
            membership_idxs = {'train_idxs': train_idxs, 'test_idxs': test_idxs}
            train_shadow_model(target_exp, membership_idxs, train, val, test, 
                    model_seeds, device, target_save_dir, args)
        elif args.which_models == 'shadow':
            assert args.attacker_access in ['shadow_dataset_model_init', 
                    'shadow_dataset', 'shadow_dataset_align_after_init',
                    'stolen_memories'], \
                f'ERROR: Invalid --attacker_access {args.attacker_access}.'
            val = Subset(dataset, idxs['shadow_val_idxs'])
            for shadow_exp in range(args.num_shadow_models):
                train_idxs, test_idxs = train_test_split(
                        idxs['shadow_train_test_idxs'], test_size=0.5, 
                        random_state=seeds['shadow_splits'][shadow_exp])
                train, test = Subset(dataset, train_idxs), \
                        Subset(dataset, test_idxs)
                if args.attacker_access == 'shadow_dataset_model_init':
                    model_seeds = ( seeds['target_models'][0, target_exp], # The attacker knows the target model's initialization.
                        seeds['shadow_models'][1, shadow_exp], # Vary the batch sampling seed.
                        seeds['shadow_models'][2, shadow_exp] # Vary the dropout seed
                    ) 
                    shadow_save_dir = os.path.join(save_dir, 'shadow_models', f'aa-{args.attacker_access}', f'exp-{target_exp}')
                elif args.attacker_access in ['shadow_dataset', 
                        'shadow_dataset_align_after_init', 'stolen_memories']:
                    model_seeds = ( seeds['shadow_models'][0, shadow_exp], # The attacker does not know the target model's initialization.
                        seeds['shadow_models'][1, shadow_exp], # Vary the batch sampling seed.
                        seeds['shadow_models'][2, shadow_exp] # Vary the dropout seed
                    )
                    if args.attacker_access == 'shadow_dataset':
                        shadow_save_dir = os.path.join(save_dir, 
                                'shadow_models', f'aa-{args.attacker_access}')
                    elif args.attacker_access == 'stolen_memories':
                        shadow_save_dir = os.path.join(save_dir, 
                                'stolen_memories', f'exp-{target_exp}',
                                f'startlayer-{args.start_layer}')
                    else:
                        shadow_save_dir = os.path.join(save_dir, 
                                'shadow_models', f'aa-{args.attacker_access}', 
                                f'exp-{target_exp}')
                else:
                    raise ValueError(f'ERROR: Invalid --attacker_access {args.attacker_access}.')
                print(f'Saving directory for the shadow model # {shadow_exp} in exp # {target_exp}', shadow_save_dir)
                # Mapping the seeds to integers as required by torch.Generator.
                model_seeds = tuple([int(ms) for ms in model_seeds])
                # The indexes of train and test records will be saved to disk together with the model.
                membership_idxs = {'train_idxs': train_idxs, 
                        'test_idxs': test_idxs}
                if args.attacker_access in ['shadow_dataset', 
                        'shadow_dataset_model_init']:
                    train_shadow_model(shadow_exp, membership_idxs, train, 
                             val, test, model_seeds, device, shadow_save_dir, 
                             args)
                elif args.attacker_access in ['shadow_dataset_align_after_init',
                        'stolen_memories']:
                    # Load the target model.
                    target_model_path = os.path.join(save_dir, 
                            'target_models', f'exp_{target_exp}_model.pickle')
                    num_classes = get_num_classes(args.dataset)
                    print(f'Loading the target model {target_exp+1}')
                    target_model, _, _, _ = load_model_and_idxs(
                            target_model_path, num_classes)
                    if args.attacker_access == 'shadow_dataset_align_after_init':
                        # Pass the target model as additional argument to the
                        # method training the shadow model, in order to align 
                        # the shadow model to the target model after 
                        # initialization.
                        #np.random.seed(seeds['shadow_models'][0, shadow_exp])
                        train_shadow_model(shadow_exp, membership_idxs, train,
                            val, test, model_seeds, device, shadow_save_dir,
                            args, target_model=target_model)
                    else: # Stolen Memories paper.
                        # Train proxy models.
                        print(f'Training proxy model #{shadow_exp}')
                        train_proxy_model(shadow_exp, target_model, 
                                args.start_layer, 
                                membership_idxs, train, 
                                val, test, model_seeds, device, 
                                shadow_save_dir, args) 
                else:
                    raise ValueError(f'Unknown --which_models={args.which_models}.')
               
        else:
            raise ValueError(f'ERROR: Invalid value for --which_models={args.which_models}')


def load_model_and_idxs(model_path, num_classes):
    with open(model_path, 'rb') as f:
        saved_model = pickle.load(f)
    model = init_model(args.architecture, num_classes)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()
    return model, saved_model['train_idxs'], saved_model['test_idxs'], \
            saved_model['best_epoch']


def run_stolen_memories_attack(args, target_exp):
    """
    Stolen memories attack against the `target_exp` model.
    """
    dataset, idxs, seeds = init_parameters(args)

    num_classes = get_num_classes(args.dataset)
    
    device = torch.device(f'cuda:{args.gpu_id}' 
            if torch.cuda.is_available() and args.use_gpu else 'cpu')

    save_dir = os.path.join(args.save_dir, args.dataset, 'attack', 
            args.architecture)

    print(f'Loading target model {target_exp+1}/{args.num_target_models}...')
    target_model_path = os.path.join(save_dir, 'target_models', f'exp_{target_exp}_model.pickle')
    target_model, target_train_idxs, target_test_idxs, _ = \
            load_model_and_idxs(target_model_path, num_classes)
    target_model = target_model.eval().to(device)
    print('Computing a linear approximation of the target model at internal',
            f'layer {args.start_layer}')

    print('Preparing the test data...')
    np.random.seed(seeds['target_splits'][target_exp])
    np.random.shuffle(target_train_idxs)
    np.random.shuffle(target_test_idxs)
    meta_model_test_idxs = np.concatenate( (target_train_idxs[-args.num_meta_model_test_records//2:], \
        target_test_idxs[-args.num_meta_model_test_records//2:]) )
    meta_model_test_mia_labels = np.concatenate( (np.full(args.num_meta_model_test_records//2, 1), \
        np.full(args.num_meta_model_test_records//2, 0)) ) 
    assert len(meta_model_test_idxs) == len(meta_model_test_mia_labels)
    print('Number of meta model test records', len(meta_model_test_idxs))
    meta_model_dataset = {'test': Subset(dataset, meta_model_test_idxs)}

    assert args.attacker_access == 'stolen_memories'

    print('Loading the proxy models...')
    shadow_models_dir = os.path.join(save_dir, 'stolen_memories', 
            f'exp-{target_exp}', f'startlayer-{args.start_layer}')
    shadow_models = []
    for shadow_model_exp in range(args.num_shadow_models):
        print(f'Loading shadow model {shadow_model_exp+1}/{args.num_shadow_models}')
        shadow_model_path = os.path.join(shadow_models_dir, f'exp_{shadow_model_exp}_model.pickle')
        shadow_model, _, _, _ = \
                load_model_and_idxs(shadow_model_path, num_classes)
        shadow_model = shadow_model.eval().to(device)
        shadow_models.append(shadow_model)

    print('Computing the average shadow model...')
    avg_shadow_model = compute_average_model(shadow_models).to(device).eval()
    
    test_loader = DataLoader(meta_model_dataset['test'], 
            batch_size=args.batch_size, shuffle=False, num_workers=0)
    scores = []
    for batch_idx, batch in enumerate(test_loader):
        images, labels = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            target_outputs = target_model(images)
            shadow_outputs = avg_shadow_model(images)
        differences = target_outputs - shadow_outputs
        sigmoids = torch.sigmoid(differences)

        for i in range(len(images)):
            scores.append(sigmoids[i][labels[i]].cpu().item())

    scores = np.array(scores)
    test_acc = accuracy_score(meta_model_test_mia_labels, scores>0.5)
    test_auc = roc_auc_score(meta_model_test_mia_labels, scores)
    print(f'Test AUC: {test_auc:.3f}, test accuracy: {test_acc:.3f}')


def run_lira_attack(args, target_exp):
    """
    Stolen memories attack against the `target_exp` model.
    """
    dataset, idxs, seeds = init_parameters(args)

    num_classes = get_num_classes(args.dataset)

    device = torch.device(f'cuda:{args.gpu_id}'
            if torch.cuda.is_available() and args.use_gpu else 'cpu')

    save_dir = os.path.join(args.save_dir, args.dataset, 'attack',
            args.architecture)

    print(f'Loading target model {target_exp+1}/{args.num_target_models}...')
    target_model_path = os.path.join(save_dir, 'target_models', f'exp_{target_exp}_model.pickle')
    target_model, target_train_idxs, target_test_idxs, _ = \
            load_model_and_idxs(target_model_path, num_classes)
    target_model = target_model.eval().to(device)

    print('Preparing the test data...')
    np.random.seed(seeds['target_splits'][target_exp])
    np.random.shuffle(target_train_idxs)
    np.random.shuffle(target_test_idxs)
    meta_model_test_idxs = np.concatenate( (target_train_idxs[-args.num_meta_model_test_records//2:], \
        target_test_idxs[-args.num_meta_model_test_records//2:]) )
    meta_model_test_mia_labels = np.concatenate( (np.full(args.num_meta_model_test_records//2, 1), \
        np.full(args.num_meta_model_test_records//2, 0)) )
    assert len(meta_model_test_idxs) == len(meta_model_test_mia_labels)
    print('Number of meta model test records', len(meta_model_test_idxs))
    test_dataset = Subset(dataset, meta_model_test_idxs)

    print('Loading the shadow models...')
    shadow_models_dir = os.path.join(save_dir, 'shadow_models', f'aa-{args.attacker_access}')
    if args.attacker_access == 'shadow_dataset_model_init':
        shadow_models_dir = os.path.join(shadow_models_dir,
                f'exp-{target_exp}')
    shadow_models = []
    for shadow_model_exp in range(args.num_shadow_models): 
        print(f'Loading shadow model {shadow_model_exp+1}/{args.num_shadow_models}')
        shadow_model_path = os.path.join(shadow_models_dir, f'exp_{shadow_model_exp}_model.pickle')
        shadow_model, _, _, _ = load_model_and_idxs(shadow_model_path, num_classes)
        shadow_model = shadow_model.eval().to(device)
        shadow_models.append(shadow_model)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
            shuffle=False, num_workers=0)

    print('Computing the logits...')
    logits_out = np.zeros((len(test_dataset), len(shadow_models)))
    logits_test = np.zeros(len(test_dataset))
    for batch_idx, batch in enumerate(test_loader):
        images, labels = batch[0].to(device), batch[1].cpu().numpy()
        start_idx = batch_idx * args.batch_size
        #end_idx = min((batch_idx+1) * args.batch_size, len(test_loader))
        # Query the shadow models.
        for si, shadow_model in enumerate(shadow_models):
            with torch.no_grad():
                outputs = F.softmax(shadow_model(images), dim=1).cpu().numpy()
            probs = [outputs[i][label] for i, label in enumerate(labels)]
            logits = [math.log(p+1e-30) - math.log(1-p + 1e-30) 
                    for p in probs]
            for i in range(len(logits)):
                logits_out[start_idx+i][si] = logits[i]
        # Query the target model.
        with torch.no_grad():
            outputs = F.softmax(target_model(images), dim=1).cpu().numpy()
        probs = [outputs[i][label] for i, label in enumerate(labels)]
        logits = [math.log(p+1e-30) - math.log(1-p + 1e-30) for p in probs]
        for i in range(len(logits)):
            logits_test[start_idx+i] = logits[i]

    print('Running the lira attack')
    means_out = np.mean(logits_out, axis=1)
    stds_out = np.std(logits_out, ddof=1, axis=1)
    scores = np.zeros(len(logits_test))
    for i in range(len(logits_test)):
        pr_out = st.norm.cdf(logits_test[i], means_out[i], stds_out[i]+1e-30)
        scores[i] = pr_out

    test_acc = accuracy_score(meta_model_test_mia_labels, scores>0.5)
    test_auc = roc_auc_score(meta_model_test_mia_labels, scores)
    print(f'Test AUC: {test_auc:.3f}, test accuracy: {test_acc:.3f}')
    
    results = {'test_auc': test_auc, 'test_acc': test_acc}

    results_save_dir = os.path.join(save_dir, 'lira', args.attacker_access)
    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)
    print('Results save directory', results_save_dir)
    with open(os.path.join(results_save_dir, f'results-{target_exp}.pickle'), 
            'wb') as f:
        pickle.dump(results, f)


def train_membership_inference_attack(args, target_exp):
    """
    Running the attack for the `target_exp`-th repetition.
    """
    dataset, idxs, seeds = init_parameters(args)

    num_classes = get_num_classes(args.dataset)
    
    device = torch.device(f'cuda:{args.gpu_id}' 
            if torch.cuda.is_available() and args.use_gpu else 'cpu')

    save_dir = os.path.join(args.save_dir, args.dataset, 'attack', args.architecture)

    # For now we only attack a single target model.
    print(f'Loading the target model {target_exp}/{args.num_target_models}...')
    target_model_path = os.path.join(save_dir, 'target_models', f'exp_{target_exp}_model.pickle')
    target_model, target_train_idxs, target_test_idxs, _ = \
            load_model_and_idxs(target_model_path, num_classes)

    print('Preparing the test data...')
    np.random.seed(seeds['target_splits'][target_exp])
    np.random.shuffle(target_train_idxs)
    np.random.shuffle(target_test_idxs)
    meta_model_test_idxs = np.concatenate( (target_train_idxs[-args.num_meta_model_test_records//2:], \
        target_test_idxs[-args.num_meta_model_test_records//2:]) )
    meta_model_test_mia_labels = np.concatenate( (np.full(args.num_meta_model_test_records//2, 1), \
        np.full(args.num_meta_model_test_records//2, 0)) ) 
    assert len(meta_model_test_idxs) == len(meta_model_test_mia_labels)
    print('Number of meta model test records', len(meta_model_test_idxs))
    meta_model_dataset = {'test': Subset(dataset, meta_model_test_idxs)}
    models = {'test': [target_model], 'val': [], 'train': []}
    mia_labels = {'test': [meta_model_test_mia_labels], 'val': [], 'train': []}

    if args.attacker_access == 'target_dataset':
        print('Preparing the validation data...')
        val_start_idx = len(target_train_idxs) - args.num_meta_model_test_records//2 - args.num_meta_model_val_records//2
        val_end_idx = len(target_train_idxs) - args.num_meta_model_test_records//2
        meta_model_val_idxs = np.concatenate( (target_train_idxs[val_start_idx:val_end_idx], \
            target_test_idxs[val_start_idx:val_end_idx]) )
        meta_model_val_mia_labels = np.concatenate( (np.full(val_end_idx - val_start_idx, 1), \
            np.full(val_end_idx - val_start_idx, 0)) ) 
        assert len(meta_model_val_idxs) == len(meta_model_val_mia_labels), \
            f'ERROR: Different number of validation records and labels {len(meta_model_val_idxs)} != {len(meta_model_val_mia_labels)}'
        print('Number of meta model validation records', len(meta_model_val_idxs))

        print('Preparing the train data...')
        if args.num_meta_model_train_records > 0:
            train_end_idx = args.num_meta_model_train_records // 2
            assert train_end_idx <= val_start_idx, f'ERROR: Too many training records, please specify a number <= {val_end_idx}.'
        else:
            train_end_idx = val_start_idx
        meta_model_train_idxs = np.concatenate( (target_train_idxs[:train_end_idx], target_test_idxs[:train_end_idx]) )
        meta_model_train_mia_labels = np.concatenate( (np.full(train_end_idx, 1), np.full(train_end_idx, 0)) ) 
        assert len(meta_model_train_idxs) == len(meta_model_train_mia_labels), \
            f'ERROR: Different number of train records and labels {len(meta_model_train_idxs)} != {len(meta_model_train_mia_labels)}'
        print('Number of meta model train records', len(meta_model_train_idxs))

        check_disjoint((meta_model_train_idxs, meta_model_val_idxs, meta_model_test_idxs))

        models.update({'val': [target_model], 'train': [target_model]})
        mia_labels.update({'val': [meta_model_val_mia_labels], 'train': [meta_model_train_mia_labels]})
    elif args.attacker_access in ['shadow_dataset_model_init', 
            'shadow_dataset', 'shadow_dataset_align_after_init']:
        print('Loading the shadow models...')
        shadow_models_dir = os.path.join(save_dir, 'shadow_models', f'aa-{args.attacker_access}')
        if args.attacker_access in ['shadow_dataset_model_init',
                'shadow_dataset_align_after_init']:
            shadow_models_dir = os.path.join(shadow_models_dir, 
                    f'exp-{target_exp}')
        shadow_train_test_idxs = idxs['shadow_train_test_idxs']  
        meta_model_val_idxs = shadow_train_test_idxs[-args.num_meta_model_val_records:]
        meta_model_train_idxs = shadow_train_test_idxs[:-args.num_meta_model_val_records]  
        print(f'Number of meta model records: train {len(meta_model_train_idxs)}, validation {len(meta_model_val_idxs)}')
        if args.alignment and args.alignment_method in \
                ['bottom_up_activation_matching',
                        'top_down_activation_matching',
                        'bottom_up_correlation_matching']:
            shadow_val_records = Subset(dataset, idxs['shadow_val_idxs'])
            activation_records = torch.cat([shadow_val_records[i][0].unsqueeze(0) for i in range(args.num_records)], dim=0)

        shadow_models, shadow_models_train_idxs = [], []
        num_trained_epochs = []
        for shadow_model_exp in range(args.num_shadow_models):
            print(f'Loading shadow model {shadow_model_exp+1}/{args.num_shadow_models}')
            shadow_model_path = os.path.join(shadow_models_dir, f'exp_{shadow_model_exp}_model.pickle')
            shadow_model, shadow_model_train_idxs, _, best_epoch = \
                    load_model_and_idxs(shadow_model_path, num_classes)
            shadow_models.append(shadow_model)
            shadow_models_train_idxs.append(shadow_model_train_idxs)
            num_trained_epochs.append(best_epoch)
            print(f'The model was early stopped at epoch {best_epoch}')
        # The model used for validating the meta-model is by default equal to
        # args.shadow_model_val_exp, except for the Texas100 dataset, for 
        # which we pick a model trained for the median number of epochs.
        if args.dataset == 'texas100':
            shadow_model_val_exp = np.argsort(num_trained_epochs)[len(num_trained_epochs)//2]
        else:
            shadow_model_val_exp = args.shadow_model_val_exp
        print(f'The validation shadow model is {shadow_model_val_exp}')
        for shadow_model_exp, (shadow_model, shadow_model_train_idxs) \
                in enumerate(zip(shadow_models, shadow_models_train_idxs)):
            if args.alignment:
                print(f'Aligning shadow model {shadow_model_exp+1}/{args.num_shadow_models} using the {args.alignment_method} method with {args.matching_method} matching.')
                if args.matching_method == 'greedy':
                    matching_method = GreedyMatching()
                elif args.matching_method == 'hungarian_algorithm':
                    matching_method = HungarianAlgorithmMatching()
                else:
                    raise ValueError(f'ERROR: Invalid --matching_method={args.matching_method}.') 
                if args.alignment_method == 'bottom_up_weight_matching':
                    shadow_model = BottomUpWeightMatchingBasedAlignment(
                            matching_method).\
                        align_layers(shadow_model, target_model)
                elif args.alignment_method == 'top_down_weight_matching':
                    shadow_model = TopDownWeightMatchingBasedAlignment(
                            matching_method).\
                        align_layers(shadow_model, target_model)
                elif args.alignment_method == 'bottom_up_activation_matching':
                    shadow_model = BottomUpActivationMatchingBasedAlignment(
                            matching_method).\
                        align_layers(shadow_model, target_model, 
                                records=activation_records)
                elif args.alignment_method == 'top_down_activation_matching':
                    shadow_model = TopDownActivationMatchingBasedAlignment(
                            matching_method).\
                        align_layers(shadow_model, target_model,
                                records=activation_records)
                elif args.alignment_method == 'bottom_up_correlation_matching':
                    shadow_model = BottomUpCorrelationMatchingBasedAlignment(
                            matching_method).\
                        align_layers(shadow_model, target_model,
                                records=activation_records)
                elif args.alignment_method == 'weight_sorting':
                    shadow_model = WeightSortingBasedAlignment().\
                            align_layers(shadow_model)
                    # Also align the target model (but do it only once).
                    if shadow_model_exp == 0:
                        models['test'][0] = WeightSortingBasedAlignment().\
                                align_layers(target_model)
                else:
                    raise ValueError(f'ERROR: Invalid --alignment_method={args.alignment_method}')
            shadow_model_train_idxs = set(shadow_model_train_idxs)
            if shadow_model_exp == shadow_model_val_exp:
                models['val'].append(shadow_model)
                meta_model_val_mia_labels = [int(idx in shadow_model_train_idxs) for idx in meta_model_val_idxs]
                mia_labels['val'].append(meta_model_val_mia_labels)
            else:
                models['train'].append(shadow_model)
                meta_model_train_mia_labels = [int(idx in shadow_model_train_idxs) for idx in meta_model_train_idxs]
                mia_labels['train'].append(meta_model_train_mia_labels)
    else:
        raise ValueError(f'ERROR: Invalid --attacker_access={args.attacker_access}.')

    meta_model_dataset.update({'val' : Subset(dataset, meta_model_val_idxs),
        'train': Subset(dataset, meta_model_train_idxs)} )

    print('Determinining automatically the architecture of the meta model.')
    image0, label0 = dataset[0]
    image0, label0 = image0.unsqueeze(0), torch.LongTensor([label0])
    criterion = torch.nn.CrossEntropyLoss()
    layer_architectures = get_architectures(image0, label0, target_model, 
            criterion, args.target_model_features, args.target_model_layers, 
            args, device)
    meta_model_architecture = 'meta-model__' + '__'.join(layer_architectures)
    print('Meta model architecture', meta_model_architecture)

    attack_method = f'aa-{args.attacker_access}'
    if args.alignment:
        attack_method += f'-align-{args.alignment_method}'
    elif args.set_based:
        attack_method += f'-set-based'
    meta_model_save_dir = os.path.join(save_dir, 'attack_results', 
            attack_method, args.target_model_features, args.target_model_layers)
    if not os.path.exists(meta_model_save_dir):
        os.makedirs(meta_model_save_dir)
    print('Meta model save directory', meta_model_save_dir)
    train_meta_model(target_exp, meta_model_dataset, models, mia_labels, 
            device, meta_model_save_dir, args, meta_model_architecture)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.experiment == 'train_models':
        train_models(args)
    elif args.experiment == 'train_meta_model':
        train_membership_inference_attack(args, args.meta_model_target_exp)
    elif args.experiment == 'stolen_memories':
        run_stolen_memories_attack(args, args.meta_model_target_exp)
    elif args.experiment == 'run_lira_attack':
        run_lira_attack(args, args.meta_model_target_exp)
    else:
        raise ValueError('ERROR: Invalid --experiment={args.experiment}')
