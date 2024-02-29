from collections import defaultdict
import configargparse
import numpy as np
import os
import pickle
from sklearn.metrics import auc, roc_curve
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.align import (GreedyMatching, 
    HungarianAlgorithmMatching,
    WeightSortingBasedAlignment, 
    BottomUpWeightMatchingBasedAlignment,
    TopDownWeightMatchingBasedAlignment,
    BottomUpActivationMatchingBasedAlignment,
    TopDownActivationMatchingBasedAlignment,
    BottomUpCorrelationMatchingBasedAlignment)
from src.dataset import get_num_classes, load_dataset
from src.logger import Logger
from src.models import CNNLarge, GenericMLP, init_model, init_optimizer


DATASET_SIZES = [50000, 25000, 12500, 6250, 2500, 1250]


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


    parser.add_argument('--use_gpu', type=str2bool, default=True)
    # Should be one of: dataset, seed_model, seed_dropout, seed_batching, seed_all, dataset_sizes.
    parser.add_argument('--varying', type=str, default='seed_model')
    parser.add_argument('--min_experiment', type=int, default=0)
    parser.add_argument('--num_experiments', type=int, default=6)
    
    # General parameters (seed, dataset, experiment).
    parser.add_argument('--dataset', type=str, default='cifar10')
    # Size of the dataset used to train the models. If equal to 0, we train on
    # the entire dataset. Otherwise, we train on datasets of this size.
    parser.add_argument('--dataset_size', type=int, default=0)
    # Directory where the results are saved.
    parser.add_argument('--save_dir', type=str, default='experiments')
    # Should be one of: `train_models` or `compute_metrics`.
    parser.add_argument('--experiment', type=str, default='train_models')
    # On how many records to compute the record alignment scores.
    parser.add_argument('--num_score_records', type=int, default=500)
    # How many records to use for the alignment.
    parser.add_argument('--num_alignment_records', type=int, default=500)
    # Which norm to use when computing the alignment scores.
    parser.add_argument('--norm', type=int, default=2)
    # Whether to align the representations and which method to use.
    parser.add_argument('--alignment', type=str2bool, default=False)
    # Should be one of: "bottom_up_weight_matching", "bottom_up_weight_sorting", 
    # "bottom_up_activation_matching", or "top_down_weight_matching".
    parser.add_argument('--alignment_method', type=str, default='bottom_up_weight_matching')
    # Should be "greedy" or "hungarian_algorithm".
    parser.add_argument('--matching_method', type=str, 
            default='hungarian_algorithm')
    
    # Model and training parameters.
    parser.add_argument('--model_config', is_config_file=True, default='configs/cifar10/cnn-large.ini')
    parser.add_argument('--architecture', type=str, default='cnn-large')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=9e-1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--transform', type=str, default='normalize')
    parser.add_argument('--max_num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_patience', type=int, default=10)
    parser.add_argument('--print_every', type=int, default=10)

    parser.add_argument('--num_val_records', type=int, default=5000)
    parser.add_argument('--eval_train', type=str2bool, default=True)

    return parser.parse_args()


def evaluate_model(model, dataloader, device, compute_auc=False):
    # To compute the AUC, set `compute_auc`` to True (only works for a
    # binary classifier.
    model.eval()
    correct = 0
    total = 0
    y_test, y_pred = [], []
    for batch in tqdm(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        if compute_auc:
            softmax = F.softmax(outputs, dim=1)
            #for i, sm in enumerate(softmax):
            #    print(sm, labels[i])
            y_pred.extend(softmax[:, 1].tolist())
            y_test.extend(labels.cpu().tolist())
            #print(y_pred, y_test)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    if compute_auc:
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        auc_score = auc(fpr, tpr)
    else:
        fpr, tpr, auc_score = None, None, None
    return acc, fpr, tpr, auc_score


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(exp, train, val, test, seeds, device, save_dir, args, 
        ref_model=None):
    save_path_prefix = os.path.join(save_dir, f'exp_{exp}')
    saved_model_path = f'{save_path_prefix}_model.pickle'

    num_classes = get_num_classes(args.dataset)

    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
            best_model = pickle.load(f)
            if best_model['train_complete']:
                print(f'The model for experiment {exp+1} is already trained.')
                model = init_model(args.architecture, num_classes).to(device)
                model.load_state_dict(best_model['model_state_dict'])
                return model

    seed_model, seed_batching, seed_dropout = seeds
    # Set the seed used to initialize the model.
    set_torch_seeds(seed_model)
    # Initialize the model using this seed.
    model = init_model(args.architecture, num_classes).to(device)
    #print(model.conv2[0].bias)
    #print(model.fc1.linear.bias[:10])

    # Align the model immediately after the initialization.
    if args.varying == 'seed_all_dataset_disjoint_align_after_init' and exp > 0:
        assert ref_model is not None
        ref_model = ref_model.to(device)
        print('Aligning the model to the target model.')
        matching_method = HungarianAlgorithmMatching()
        model = TopDownWeightMatchingBasedAlignment(matching_method).\
                align_layers(model, ref_model)

    # Reset the seed for dropout.
    set_torch_seeds(seed_dropout)

    # Set the seed for batching.
    g = torch.Generator()
    g.manual_seed(seed_batching)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, 
            num_workers=0, generator=g)

    if args.eval_train:
        seq_train = train
    else:
        seq_train = Subset(train, range(len(val)))
    seq_train_loader = DataLoader(seq_train, 
                batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, 
            num_workers=0)

    optimizer = init_optimizer(model, args.optimizer, args.learning_rate, 
            args.momentum, args.weight_decay)
    learning_rate_scheduler = ExponentialLR(optimizer, gamma=0.5)

    criterion = nn.CrossEntropyLoss().to(device)

    logger = Logger(exp, args.print_every, save_path_prefix)
    best_val_acc = 0
    early_stopping_count = 0
    it = 1
    for epoch in range(args.max_num_epochs + 1):
        model.train()
        train_iter = iter(train_loader)
        # Do not train for epoch 0, just evaluate the model.
        while True and epoch > 0:
            optimizer.zero_grad()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            #print(inputs[:, 0, 0, 0])
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            logger.log_loss(loss.item(), epoch, it)

            loss.backward()
            optimizer.step()
            it += 1

        # Evaluate the model's accuracy on the training and validation datasets.
        start_time = time.time()
        epoch_summary = f'End of epoch {epoch}. Accuracy on '
        print('Evaluating the model on the training data.')
        train_acc, _, _, _ = evaluate_model(model, seq_train_loader, device)
        epoch_summary += f'train: {train_acc:.1%} '
        print('Evaluating the model on the validation data.')
        val_acc, _, _, _ = evaluate_model(model, val_loader, device)
        print(epoch_summary + f'validation: {val_acc:.1%}. Elapsed time: {time.time()-start_time:.2f} secs')
        logger.log_accuracy(train_acc, val_acc)

        if val_acc > best_val_acc:
            print(f'The validation accuracy has improved from {best_val_acc:.1%} to {val_acc:.1%}. Saving the parameters to disk.')
            best_val_acc = val_acc
            
            with open(saved_model_path, 'wb') as f:
                pickle.dump({'model_state_dict': model.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_complete': False}, f)
        else:
            early_stopping_count += 1
            if early_stopping_count == args.num_epochs_patience:
                early_stopping_count = 0
                learning_rate_scheduler.step()
                new_learning_rate = optimizer.param_groups[0]['lr']
                print(f'New learning rate: {new_learning_rate}')
                if new_learning_rate < args.min_learning_rate:
                    print(f'Stopping the training because the learning rate is lower than {args.min_learning_rate}.')
                    break

    print('End of training. Loading the best model to mark the training as complete.')
    with open(saved_model_path, 'rb') as f:
        best_model = pickle.load(f)
        best_model['train_complete'] = True
        best_model['seeds'] = seeds
        model = init_model(args.architecture, num_classes).to(device)
        model.load_state_dict(best_model['model_state_dict'])
        
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc, test_fpr, test_tpr, test_auc = evaluate_model(model, test_loader, device)
        best_model.update({'test_acc': test_acc, 'test_fpr': test_fpr, 'test_tpr': test_tpr, 'test_auc': test_auc})
        train_acc, val_acc = best_model['train_acc'], best_model['val_acc']
        print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
    with open(saved_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    return model


def compute_correlations_ref_others(activations_ref, activations_others, 
        device='cpu'):
    """
    activations_ref:  list containing activations for one layer of the 
        reference model.
        The size of each element is B x N x other dimensions, where B is 
        the number of records we compute activations over, N is the number 
        of units (feature maps for CNNs or neurons for MLPs), and the 
        number of other dimensions is 2 for feature maps and 1 for MLPs.
    activations_others: list of activations similar to activations_ref, 
        with one element per model.

    For the i-th model, we measure its alignment with the reference model.
    For each layer, we compute the correlations between each unit in 
    activations_ref and the corresponding unit in activations_others[i]. 
    Returns the average correlation over all the units.
    """
    num_layers = len(activations_ref)
    activations_ref = [torch.cat(activations_ref[l], dim=0).to(device) 
            for l in range(num_layers)]
    #print(activations_ref[0].size())
    all_corrs = []
    for activations_other in activations_others:
        #print('Other model')
        activations_other = [torch.cat(activations_other[l], dim=0).to(device)
                for l in range(num_layers)]
        #print('Other model size', activations_other[0].size())
        corrs = []
        for layer_a1, layer_a2 in zip(activations_ref, activations_other):
            #print('Layer', layer_a1.size(), layer_a2.size())
            layer_corrs = compute_pairwise_correlations(layer_a1, layer_a2, 
                    device)
            corrs.append(layer_corrs)
            #print(layer_corrs.size())
        all_corrs.append(corrs)
    return all_corrs


def compute_pairwise_correlations(a1, a2, device):
    assert a1.size() == a2.size()
    B, N = a1.size(0), a1.size(1)
    a1, a2 = a1.view(B, N, -1), a2.view(B, N, -1)
    F = a1.size(2)
    num_features = min(50, F)
    features = np.random.randint(F, size=num_features)
    #print(a1.size(), a1.std(dim=0).size())
    a1 += torch.FloatTensor(np.random.uniform(1e-8, size=a1.size())).to(device)
    a2 += torch.FloatTensor(np.random.uniform(1e-8, size=a2.size())).to(device)
    #print(a1.size(), a2.size())
    a1 = (a1 - a1.mean(dim=0)) / a1.std(dim=0)
    a2 = (a2 - a2.mean(dim=0)) / a2.std(dim=0)
    #print(a1.mean(dim=0)[0, 0], a1.std(dim=0)[0, 0], a2.mean(dim=0)[0, 0])
    corrs = torch.zeros(size=(N, num_features))
    for i in range(N):
        for k, f in enumerate(features):
            # Compute the correlation manually (much faster).
            a12 = torch.dot(a1[:, i, f], a2[:, i ,f]) / B
            #print(a12)
            corrs[i, k] = a12
    return corrs.mean().item()


def compute_metrics_controlled_randomness(args):
    # Load the dataset.
    dataset = load_dataset(args.dataset, args.transform, dataset_size=0, seed=None)

    dataset_size = len(dataset['train_and_test']) if args.dataset_size == 0 else args.dataset_size
    models_dir = os.path.join(args.save_dir, args.dataset, 
            'controlled_randomness', args.architecture, 
            f'dsize-{dataset_size}', args.varying)
    num_classes = get_num_classes(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    #device = 'cpu'

    if args.alignment and args.alignment_method in \
            ['bottom_up_activation_matching', 'top_down_activation_matching',
                    'bottom_up_correlation_matching']:
        if dataset['val'] is None: # Purchase dataset.
            activation_records = torch.cat(\
                [dataset['train_and_test'][-i][0].unsqueeze(0) 
                    for i in range(args.num_alignment_records)], 
                dim=0).to(device)
        else:
            activation_records = torch.cat(\
                [dataset['val'][-i][0].unsqueeze(0) 
                    for i in range(args.num_alignment_records)], 
                dim=0).to(device)

    models = []
    for exp in range(args.num_experiments):
        models_path = os.path.join(models_dir, f'exp_{exp}_model.pickle')
        if not os.path.exists(models_path):
            continue
        with open(models_path, 'rb') as f:
            saved_model = pickle.load(f)
        model = init_model(args.architecture, num_classes, verbose=False)
        model.load_state_dict(saved_model['model_state_dict'])
        model = model.to(device)
        model.eval()
        if exp == 0:
            ref_model = model
            # Retrieve the model seed.
            #if 'seeds' in saved_model:
            #    ref_model.seed = saved_model['seeds'][0]
        else:
            if args.alignment:
                if args.matching_method == 'greedy':
                    matching_method = GreedyMatching()
                elif args.matching_method == 'hungarian_algorithm':
                    matching_method = HungarianAlgorithmMatching()
                else:
                    raise ValueError(f'ERROR: Invalid --matching_method={args.matching_method}.')
                if args.alignment_method == 'bottom_up_weight_matching':
                    model = BottomUpWeightMatchingBasedAlignment(
                            matching_method).align_layers(model, ref_model)
                elif args.alignment_method == 'top_down_weight_matching':
                    model = TopDownWeightMatchingBasedAlignment(
                            matching_method).align_layers(model, ref_model)
                elif args.alignment_method == 'bottom_up_activation_matching':
                    model = BottomUpActivationMatchingBasedAlignment(
                            matching_method).align_layers(model, ref_model, 
                                    records=activation_records)
                elif args.alignment_method == 'top_down_activation_matching':
                    model = TopDownActivationMatchingBasedAlignment(
                            matching_method).align_layers(model, ref_model,
                                    records=activation_records)
                elif args.alignment_method == 'bottom_up_correlation_matching':
                    model = BottomUpCorrelationMatchingBasedAlignment(
                            matching_method, device).\
                                    align_layers(model, ref_model,
                                    records=activation_records)
                elif args.alignment_method == 'weight_sorting':
                    model = WeightSortingBasedAlignment().align_layers(model)
                    if exp == 1: # Also align the reference model (but do it only once).
                        ref_model = WeightSortingBasedAlignment().align_layers(ref_model)
                else:
                    raise ValueError(f'ERROR: Invalid --alignment_method={args.alignment_method}')
            models.append(model)

    assert ref_model, 'ERROR: No reference model found (corresponding to exp=0).'
    assert len(models) > 0, 'ERROR: No models found (corresponding to exp>0).'

    np.random.seed(0)
    ref_weights = get_weights_by_layer(ref_model)

    # Comparison with the reference model's own initialization (before training).
    #if hasattr(ref_model, 'seed'):
    #    set_torch_seeds(ref_model.seed)
    #    ref_model_init = init_model(args.architecture, num_classes, 
    #            verbose=False)
    #    ref_model_init = ref_model_init.to(device)
    #    ref_init_weights = get_weights_by_layer(ref_model_init)
    #    wms_init = compute_distances(ref_weights, ref_init_weights, args.norm)
    #    print('Weight misalignment scores for the reference model and its initialization',
    #            get_results_string( (wms_init, ) ) ) 

    # Baseline: permutation.
    permuted_weight_alignment_scores = []
    for _ in range(len(models)):
        permuted_weights = get_weights_by_layer(ref_model, permute=True)
        permuted_weight_alignment_scores.append(
            compute_distances(ref_weights, permuted_weights, args.norm))
    mean_pwas_per_layer = np.mean(permuted_weight_alignment_scores, axis=0)
    std_pwas_per_layer = np.std(permuted_weight_alignment_scores, axis=0, ddof=1)
    print(f'\nWeight misalignment score (Random permutation) for', 
        get_results_string( (mean_pwas_per_layer, std_pwas_per_layer) ))

    weight_alignment_scores = []
    for model in models:
        weights = get_weights_by_layer(model)
        weight_alignment_scores.append(compute_distances(ref_weights, weights, args.norm))
    weight_alignment_scores = np.array(weight_alignment_scores)
    if args.varying == 'dataset_sizes':
        for i in range(len(models)):
            print(f'Weight misalignment score w.r.t. dataset_size={DATASET_SIZES[i+1]} for',
                get_results_string( (weight_alignment_scores[i], ) ))
    else:
        mean_was_per_layer = np.mean(weight_alignment_scores, axis=0)
        std_was_per_layer = np.std(weight_alignment_scores, axis=0, ddof=1)
        print(f'Weight misalignment score w.r.t. {args.varying} for', 
            get_results_string( (mean_was_per_layer, std_was_per_layer) ))

    np.random.seed(0)
    if dataset['val'] is not None:
        records = Subset(dataset['val'], 
                np.random.permutation(len(dataset['val']))[:args.num_score_records])
    else:
        total_records = len(dataset['train_and_test'])
        records = Subset(dataset['train_and_test'], 
                np.arange(total_records-10000, total_records-10000+args.num_score_records))
    dataloader = DataLoader(records, shuffle=False, batch_size=1, num_workers=0)

    if isinstance(model, CNNLarge):
        num_layers = 4
    elif isinstance(model, GenericMLP):
        num_layers = len(model.layer_sizes) - 1
    else:
        raise ValueError(f'ERROR: Invalid model type {type(model)}.')
    permuted_record_alignment_scores = np.zeros((len(records), len(models), num_layers))
    all_ref_activations = [ [] for _ in range(num_layers)]
    all_permuted_activations = [ [ [] for _ in range(num_layers)] for _ in range(len(models))]
    for ri, (record, _) in enumerate(dataloader):
        ref_activations = compute_activations(ref_model, record, device)
        for li in range(num_layers):
            all_ref_activations[li].append(ref_activations[li])
        for mi in range(len(models)):
            # The seeding for the shuffling is always the same given a model.
            permuted_activations = compute_activations(ref_model, record, device, 
                    permute=True, seed=mi)
            for li in range(num_layers):
                all_permuted_activations[mi][li].append(permuted_activations[li])
            permuted_record_alignment_scores[ri][mi] = np.array(
                    compute_distances(ref_activations, permuted_activations, 
                        args.norm))
    permuted_record_alignment_scores = np.mean(permuted_record_alignment_scores,
            axis=0)
    mean_pras_per_layer = np.mean(permuted_record_alignment_scores, axis=0)
    std_pras_per_layer = np.std(permuted_record_alignment_scores, axis=0, 
            ddof=1)
    print('\nActivation misalignment score (Random Permutation) for', 
        get_results_string( (mean_pras_per_layer, std_pras_per_layer) ))
    
    record_alignment_scores = np.zeros((len(records), len(models), num_layers))
    all_activations = [ [ [] for _ in range(num_layers)] for _ in range(len(models))]
    for ri, (record, _) in enumerate(dataloader):
        ref_activations = compute_activations(ref_model, record, device)
        for mi, model in enumerate(models):
            activations = compute_activations(model, record, device)
            for li in range(num_layers):
                all_activations[mi][li].append(activations[li])
            record_alignment_scores[ri][mi] = np.array(
                    compute_distances(ref_activations, activations, args.norm))
    record_alignment_scores = np.mean(record_alignment_scores, axis=0)
    if args.varying == 'dataset_sizes':
        for i in range(len(models)):
            print(f'Activation misalignment score w.r.t. dataset_size={DATASET_SIZES[i+1]} for',
                get_results_string( (record_alignment_scores[i], ) ))
    else:
        mean_ras_per_layer = np.mean(record_alignment_scores, axis=0)
        std_ras_per_layer = np.std(record_alignment_scores, axis=0, ddof=1)
        print(f'Activation misalignment score w.r.t. {args.varying} for', 
            get_results_string( (mean_ras_per_layer, std_ras_per_layer) ))
    
    permuted_act_corrs = compute_correlations_ref_others(all_ref_activations,
            all_permuted_activations, device)
    mean_permuted_act_corrs = np.mean(permuted_act_corrs, axis=0)
    std_permuted_act_corrs = np.std(permuted_act_corrs, axis=0, ddof=1)
    print('\nCorrelation between activations (Random permutation) for ',
        get_results_string( (mean_permuted_act_corrs, std_permuted_act_corrs) ))

    act_corrs = compute_correlations_ref_others(all_ref_activations,
        all_activations, device)
    mean_act_corrs = np.mean(act_corrs, axis=0)
    std_act_corrs = np.std(act_corrs, axis=0, ddof=1)
    print(f'Correlation between activations w.r.t. {args.varying} for',
            get_results_string( (mean_act_corrs, std_act_corrs) ))


def compute_metrics_controlled_randomness_celeba(args):
    # Load the CelebA dataset.
    dataset = load_dataset(args.dataset, args.transform, dataset_size=0, 
            seed=None)

    dataset_size = len(dataset['train_and_test']) if args.dataset_size == 0 else args.dataset_size
    dataset_names = ['celeba-old', 'celeba', 'celeba', 'celeba-old']
    varying = ['dataset_disjoint', 'dataset_disjoint', 
            'seed_all_dataset_disjoint', 'seed_all_dataset_disjoint']
    models_dirs = [os.path.join(args.save_dir, dataset_name, 
            'controlled_randomness', args.architecture, 
            f'dsize-{dataset_size}', v) 
            for dataset_name, v in zip(dataset_names, varying)]
    num_experiments = [2, 4, 6, 2]
    num_classes = get_num_classes(args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    #device = 'cpu'
    
    models = {'celeba': defaultdict(list), 'celeba-old': defaultdict(list)}
    for di in range(len(dataset_names)):
        dataset_name = dataset_names[di]
        v = varying[di]
        for exp in range(num_experiments[di]):
            models_path = os.path.join(models_dirs[di], 
                    f'exp_{exp}_model.pickle')
            if not os.path.exists(models_path):
                continue
            with open(models_path, 'rb') as f:
                saved_model = pickle.load(f)
            model = init_model(args.architecture, num_classes, verbose=False)
            model.load_state_dict(saved_model['model_state_dict'])
            model = model.to(device)
            model.eval()
            models[dataset_name][v].append(model)
        assert len(models[dataset_name][v]) > 0, \
                f'ERROR: No models found for {dataset_name} and {v}.'

    # The reference model is always celeba-old trained with the first seed.
    np.random.seed(0)
    ref_model = models['celeba-old']['dataset_disjoint'][0]
    ref_weights = get_weights_by_layer(ref_model)
    # Baseline: permutation.
    permuted_weight_alignment_scores = []
    for _ in range(5):
        permuted_weights = get_weights_by_layer(ref_model, permute=True)
        permuted_weight_alignment_scores.append(
            compute_distances(ref_weights, permuted_weights, args.norm))
    mean_pwas_per_layer = np.mean(permuted_weight_alignment_scores, axis=0)
    std_pwas_per_layer = np.std(permuted_weight_alignment_scores, axis=0, ddof=1)
    print(f'\nWeight misalignment score (random permutation) \n for', 
        get_results_string( (mean_pwas_per_layer, std_pwas_per_layer) ))

    # Comparison with a model trained using same seed and a disjoint subset of 
    # old faces.
    same_old = models['celeba-old']['dataset_disjoint'][1]
    same_old_weights = get_weights_by_layer(same_old)
    same_old_weight_alignment_scores = compute_distances(ref_weights, 
            same_old_weights, args.norm)
    print(f'Weight misalignment score w.r.t. old+same seed+disjoint dataset \n',
            get_results_string( (same_old_weight_alignment_scores, ) ))

    diff_old = models['celeba-old']['seed_all_dataset_disjoint'][1]
    diff_old_weights = get_weights_by_layer(diff_old)
    diff_old_weight_alignment_scores = compute_distances(ref_weights,
            diff_old_weights, args.norm)
    print(f'Weight misalignment score w.r.t. diff seed+disjoint old \n',
            get_results_string( (diff_old_weight_alignment_scores, ) ))

    # Comparison with a model trained with same randomness but random subset 
    # (predominantly young faces).
    same_random_weight_alignment_scores = []
    for model in models['celeba']['dataset_disjoint'][1:]:
        weights = get_weights_by_layer(model)
        same_random_weight_alignment_scores.append(compute_distances(
            ref_weights, weights, args.norm))
    same_random_weight_alignment_scores = np.array(
            same_random_weight_alignment_scores)
    mean_was_per_layer = np.mean(same_random_weight_alignment_scores, axis=0)
    std_was_per_layer = np.std(same_random_weight_alignment_scores, axis=0,
            ddof=1)
    print(f'Weight misalignment score w.r.t. random+same seed \n', 
            get_results_string( (mean_was_per_layer, std_was_per_layer) ))

    # Comparison with five models trained using different randomness and random
    # subsets (predominantly young faces).
    diff_random_weight_alignment_scores = []
    for model in models['celeba']['seed_all_dataset_disjoint'][1:]:
        weights = get_weights_by_layer(model)
        diff_random_weight_alignment_scores.append(
                compute_distances(ref_weights, weights, args.norm))
    diff_random_weight_alignment_scores = np.array(
            diff_random_weight_alignment_scores)
    mean_was_per_layer = np.mean(diff_random_weight_alignment_scores, axis=0)
    std_was_per_layer = np.std(diff_random_weight_alignment_scores, axis=0, 
            ddof=1)
    print(f'Weight misalignment score w.r.t. random+different seed, \n for', 
            get_results_string( (mean_was_per_layer, std_was_per_layer) ))

   
def get_results_string(results):
    stdout = ''
    if len(results) == 2:
        means, stds = results
        for i, (mean, std) in enumerate(zip(means, stds)):
            stdout += f'layer {i+1}: {mean:.2f} ({std:.2f})'
            if i < len(means) - 1:
                stdout += '; '
    else:
        assert len(results) == 1
        for i, r in enumerate(results[0]):
            stdout += f'layer {i+1}: {r:.2f}'
            if i < len(results[0]) - 1:
                stdout += '; '
    return stdout


def get_weights_by_layer(model, permute=False):
    """
    Extracts the weights of a CNNLarge or GenericMLP model, layer by layer.
    """
    weights = []
    
    if isinstance(model, CNNLarge):
        w1, b1 = model.conv1[0].weight, model.conv1[0].bias
        if permute:
            shuffled_idxs1 = np.random.permutation(len(w1))
            w1, b1 = w1[shuffled_idxs1], b1[shuffled_idxs1]
        weights.append(torch.cat((w1.flatten(), b1)))

        w2, b2 = model.conv2[0].weight, model.conv2[0].bias
        if permute:
            shuffled_idxs2 = np.random.permutation(len(w2))
            w2, b2 = w2[shuffled_idxs2], b2[shuffled_idxs2]
        weights.append(torch.cat((w2.flatten(), b2)))

        w3, b3 = model.fc1.linear.weight, model.fc1.linear.bias
        if permute:
            shuffled_idxs3 = np.random.permutation(len(w3))
            w3, b3 = w3[shuffled_idxs3], b3[shuffled_idxs3]
        weights.append(torch.cat((w3.flatten(), b3)))

        w4, b4 = model.fc2.linear.weight, model.fc2.linear.bias
        if permute:
            shuffled_idxs4 = np.random.permutation(len(w4))
            w4, b4 = w4[shuffled_idxs4], b4[shuffled_idxs4]
        weights.append(torch.cat((w4.flatten(), b4)))
    elif isinstance(model, GenericMLP):
        num_layers = len(model.layer_sizes) - 1
        for i in range(num_layers):
            linear_layer =  getattr(model, f'fc{i+1}').linear
            w, b = linear_layer.weight, linear_layer.bias
            if permute:
                shuffled_idxs = np.random.permutation(len(w))
                w, b = w[shuffled_idxs], b[shuffled_idxs]
            weights.append(torch.cat( (w.flatten(), b) ))
    else:
        raise ValueError(f'ERROR: Invalid model type {type(model)}.')

    return weights


@torch.no_grad()
def compute_distances(list1, list2, norm=2):
    """
    Computes the list of distances between pairs of elements in list1 and list2.
    """
    assert len(list1) == len(list2), 'ERROR: Different number of inputs in each vector.'
    distances = []
    for i in range(len(list1)):
        distance = float(torch.norm(list1[i] - list2[i], p=norm).cpu())
        if norm == 1: # Normalize the distance by the number of values.
            distance = distance / len(list1[i])
        distances.append(distance)
    return distances


@torch.no_grad()
def compute_activations(model, x, device, permute=False, seed=0):
    np.random.seed(seed)
    # x is of size 1 x num_channels x size1 x size2.
    model.eval()
    x = x.to(device)

    if isinstance(model, CNNLarge):
        x1 = model.conv1(x)
        x2 = model.conv2(x1)
        x3 = model.fc1(model.flatten(x2))
        x4 = F.softmax(model.fc2(x3), dim=1)
        if permute:
            #print('Before', x1.size(), x2.size(), x3.size(), x4.size(), x4)
            x1 = x1[:, np.random.permutation(x1.size(1))]
            x2 = x2[:, np.random.permutation(x2.size(1))]
            x3 = x3[:, np.random.permutation(x3.size(1))]
            x4 = x4[:, np.random.permutation(x4.size(1))]
            #print('After', x1.size(), x2.size(), x3.size(), x4.size(), x4)
            #print(x1.numel(), x2.numel(), x3.numel(), x4.numel())
        return x1, x2, x3, x4
    elif isinstance(model, GenericMLP):
        activations = []
        num_layers = len(model.layer_sizes) - 1
        x_layer = x
        for i in range(num_layers):
            x_layer = getattr(model, f'fc{i+1}')(x_layer)
            if i == num_layers - 1:
                x_layer = F.softmax(x_layer, dim=1)
            activations.append(x_layer)
        if permute:
            for i in range(num_layers):
                activations[i] = activations[i][:, np.random.permutation(activations[i].size(1))]
        return activations
    else:
        raise ValueError(f'ERROR: Invalid model type {type(model)}.')


def train_models_controlled_randomness(args):
    # Load the dataset.
    dataset = load_dataset(args.dataset, args.transform, dataset_size=0, seed=None)

    seeds_model = list(range(100))
    seeds_batching = list(range(100, 200, 1))
    seeds_dropout = list(range(200, 300, 1))
    seeds_dataset = list(range(300, 400, 1))

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    if dataset['val'] is not None:
        all_train = dataset['train_and_test']       
        val = Subset(dataset['val'], np.arange(len(dataset['val'])//2))
        test = Subset(dataset['val'], np.arange(len(dataset['val'])//2, len(dataset['val']), 1))
    else:
        num_records = len(dataset['train_and_test'])
        all_train = Subset(dataset['train_and_test'], np.arange(num_records-2*args.num_val_records))
        val = Subset(dataset['train_and_test'], np.arange(num_records-2*args.num_val_records, num_records-args.num_val_records))
        test = Subset(dataset['train_and_test'], np.arange(num_records-args.num_val_records, num_records))

    dataset_size = len(all_train) if args.dataset_size == 0 else args.dataset_size
    save_dir = os.path.join(args.save_dir, args.dataset, 'controlled_randomness', args.architecture, f'dsize-{dataset_size}', args.varying)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.varying == 'dataset_disjoint':
        # Partitioning the dataset into 4 disjoint parts.
        assert dataset_size < len(all_train) // 2, \
            f'ERROR: Cannot train models on disjoint datasets because of too large --dataset_size.'
        seeds = (seeds_model[0], seeds_batching[0], seeds_dropout[0])
        num_experiments = min(len(all_train) // dataset_size, args.num_experiments)
        np.random.seed(seeds_dataset[0])
        random_idxs = np.random.permutation(len(all_train))
        for exp in range(num_experiments):
            start_idx = exp * dataset_size
            end_idx = min((exp + 1) * dataset_size, len(all_train))
            print(f'Training the model on (shuffled) dataset records ranging from {start_idx} to {end_idx}.')
            train = Subset(all_train, random_idxs[start_idx:end_idx])
            train_model(exp, train, val, test, seeds, device, save_dir, args)
    elif args.varying == 'dataset_overlapping':
        seeds = (seeds_model[0], seeds_batching[0], seeds_dropout[0])
        for exp in range(args.min_experiment, args.min_experiment+args.num_experiments, 1):
            if args.dataset_size > 0:
                np.random.seed(seeds_dataset[exp])
                random_idxs = np.random.permutation(len(all_train))[:dataset_size]
                train = Subset(all_train, random_idxs)
            else:
                raise ValueError('ERROR: Cannot have --dataset_size=0 for the dataset_overlapping experiment')
            train_model(exp, train, val, test, seeds, device, save_dir, args)
    elif args.varying == 'dataset_sizes':
        seeds = (seeds_model[0], seeds_batching[0], seeds_dropout[0])
        for exp in range(len(DATASET_SIZES)):
            print(f'Training dataset size: {DATASET_SIZES[exp]}')
            #np.random.seed(seeds_dataset[0])
            #random_idxs = np.random.permutation(len(all_train))[:dataset_sizes[exp]]
            random_idxs = np.arange(DATASET_SIZES[exp])
            train = Subset(all_train, random_idxs)
            train_model(exp, train, val, test, seeds, device, save_dir, args)
    else:
        if args.varying == 'seed_model':
            seeds = [(seeds_model[i], seeds_batching[0], seeds_dropout[0]) 
                    for i in range(len(seeds_model))]
        elif args.varying == 'seed_batching':
            seeds = [(seeds_model[0], seeds_batching[i], seeds_dropout[0]) 
                    for i in range(len(seeds_batching))]
        elif args.varying == 'seed_dropout':
            seeds = [(seeds_model[0], seeds_batching[0], seeds_dropout[i]) 
                    for i in range(len(seeds_dropout))]
        elif args.varying == 'seed_all':
            seeds = [(seeds_model[i], seeds_batching[i], seeds_dropout[i]) 
                    for i in range(len(seeds_model))]
        elif args.varying == 'seed_all_dataset_disjoint':
            seeds = [(seeds_model[i], seeds_batching[i], seeds_dropout[i]) 
                    for i in range(len(seeds_model))]
        elif args.varying == 'seed_all_dataset_disjoint_align_after_init':
            # Same as above.
            seeds = [(seeds_model[i], seeds_batching[i], seeds_dropout[i])
                    for i in range(len(seeds_model))]
        elif args.varying == 'seed_batching_dropout_dataset_disjoint':
            seeds = [(seeds_model[0], seeds_batching[i], seeds_dropout[i]) 
                    for i in range(len(seeds_model))]
        else:
            raise ValueError(f'ERROR: Invalid `--varying` parameter {args.varying}.')

        if args.varying in ['seed_model', 'seed_batching', 'seed_dropout', 'seed_all']:
            for exp in range(args.min_experiment, args.min_experiment+args.num_experiments, 1):
                if args.dataset_size > 0:                    
                    np.random.seed(seeds_dataset[0])
                    random_idxs = np.random.permutation(len(all_train))[:dataset_size]
                    train = Subset(all_train, random_idxs)
                else:
                    train = all_train
                print(f'Starting experiment {exp+1}/{args.min_experiment+args.num_experiments}')
                train_model(exp, train, val, test, seeds[exp], device, 
                        save_dir, args)
        elif args.varying in ['seed_all_dataset_disjoint', 'seed_batching_dropout_dataset_disjoint', 'seed_all_dataset_disjoint_align_after_init']:  
            assert dataset_size < len(all_train) // 2, \
                f'ERROR: Cannot train models on disjoint datasets because of too large --dataset_size.'
            num_experiments = min(len(all_train) // dataset_size, 
                    args.num_experiments)
            np.random.seed(seeds_dataset[0])
            random_idxs = np.random.permutation(len(all_train))
            ref_model = None # Useful for `seed_all_dataset_disjoint_align_after_init`.
            for exp in range(num_experiments):
                start_idx = exp * dataset_size
                end_idx = min((exp + 1) * dataset_size, len(all_train))
                print(f'Training the model on (shuffled) dataset records ranging from {start_idx} to {end_idx}/{len(all_train)}.')
                train = Subset(all_train, random_idxs[start_idx:end_idx])
                if args.varying == 'seed_all_dataset_disjoint_align_after_init':
                    if exp == 0:
                        ref_model = train_model(exp, train, val, test, 
                                seeds[exp], device, save_dir, args)
                    else:
                        assert ref_model is not None
                        train_model(exp, train, val, test, seeds[exp], 
                                device, save_dir, args, ref_model=ref_model)
                else:
                    train_model(exp, train, val, test, seeds[exp], device, 
                            save_dir, args)                    
        else:
            raise ValueError(f'ERROR: Invalid value for the --varying parameter {args.varying}.')



if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.experiment == 'train_models':
        train_models_controlled_randomness(args)
    elif args.experiment == 'compute_metrics':
        if args.dataset == 'celeba':
            compute_metrics_controlled_randomness_celeba(args)
        elif args.dataset == 'celeba-old':
            raise ValueError('ERROR: For a controlled evaluation on the CelebA dataset, set --dataset to `celeba`.')
        else:
            compute_metrics_controlled_randomness(args)
    else:
        raise ValueError('ERROR: Invalid --experiment={args.experiment}')
