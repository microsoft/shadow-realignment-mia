import copy
#from opacus import PrivacyEngine
#from opacus.validators import ModuleValidator
#from opacus.utils.batch_memory_manager import BatchMemoryManager
import os
import pickle
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm

from src.align import (HungarianAlgorithmMatching,
        TopDownWeightMatchingBasedAlignment)
from src.dataset import get_num_classes
from src.logger import Logger
from src.models import init_model, init_optimizer
from src.resnet import ResNet


def evaluate_shadow_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    for batch in tqdm(dataloader):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc


def set_torch_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_shadow_model(exp, membership_idxs, train, val, test, seeds, device, save_dir, args, eval_train=True, target_model=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path_prefix = os.path.join(save_dir, f'exp_{exp}')
    saved_model_path = f'{save_path_prefix}_model.pickle'

    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
            best_model = pickle.load(f)
            if best_model['train_complete']:
                print(f'The model for experiment {exp+1} is already trained.')
                print(f'Number of epochs until the best model was found:', best_model['best_epoch'])
                train_acc, val_acc, test_acc = best_model['train_acc'], best_model['val_acc'], best_model['test_acc']
                print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
                return None

    seed_model, seed_batching, seed_dropout = seeds
    # Set the seed used to initialize the model.
    set_torch_seeds(seed_model)
    # Initialize the model using this seed.
    num_classes = get_num_classes(args.dataset)
    model = init_model(args.architecture, num_classes).to(device)
    #print(model.conv2[0].bias)

    # Align the model immediately after the initialization.
    if args.attacker_access == 'shadow_dataset_align_after_init':
        assert target_model is not None
        target_model = target_model.to(device)
        print('Aligning the model to the target model.')
        matching_method = HungarianAlgorithmMatching()
        model = TopDownWeightMatchingBasedAlignment(matching_method).\
                align_layers(model, target_model)

    # Reset the seed for dropout.
    set_torch_seeds(seed_dropout)

    # Set the seed for batching.
    g = torch.Generator()
    g.manual_seed(seed_batching) 

    print('Training set size', len(train))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    if not eval_train:
        seq_train = Subset(train, range(len(val)))
    else:
        seq_train = train
    seq_train_loader = DataLoader(seq_train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = init_optimizer(model, args.optimizer, args.learning_rate, args.momentum, args.weight_decay)
    gamma = 0.9 if isinstance(model, ResNet) else 0.5
    learning_rate_scheduler = ExponentialLR(optimizer, gamma=gamma)

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
        #if eval_train:
        train_acc = evaluate_shadow_model(model, seq_train_loader, device)
        epoch_summary += f'train: {train_acc:.1%} '
        #else:
        #    train_acc = 0
        print('Evaluating the model on the validation data.')
        val_acc = evaluate_shadow_model(model, val_loader, device)
        print(epoch_summary + f'validation: {val_acc:.1%}. Elapsed time: {time.time()-start_time:.2f} secs')
        logger.log_accuracy(train_acc, val_acc)

        if args.limit_overfitting:
            print('Train-val gap:', train_acc - val_acc)
            if train_acc - val_acc > 0.05:
                break
        if args.select_best_model:
            if val_acc > best_val_acc:
                print(f'The validation accuracy has improved from {best_val_acc:.1%} to {val_acc:.1%}. Saving the parameters to disk.')
                best_val_acc = val_acc
            
                with open(saved_model_path, 'wb') as f:
                    pickle.dump({'model_state_dict': model.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'train_complete': False,
                        'best_epoch': epoch}, f)
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
        else: # Save the model after each epoch (and the final model is the model from the last epoch).
            if epoch > 0:
                learning_rate_scheduler.step()
                new_learning_rate = optimizer.param_groups[0]['lr']
                print(f'New learning rate: {new_learning_rate}')
            with open(saved_model_path, 'wb') as f:
                pickle.dump({'model_state_dict': model.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'train_complete': False,
                    'best_epoch': epoch}, f)

    print('End of training. Loading the best model to mark the training as complete.')
    with open(saved_model_path, 'rb') as f:
        best_model = pickle.load(f)
        best_model['train_complete'] = True
        model = init_model(args.architecture, num_classes).to(device)
        model.load_state_dict(best_model['model_state_dict'])
        
        if 'best_epoch' in best_model:
            print('Best epoch: ', best_model['best_epoch'])
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc = evaluate_shadow_model(model, test_loader, device)
        best_model.update({'test_acc': test_acc, 
            'train_idxs': membership_idxs['train_idxs'], 
            'test_idxs': membership_idxs['test_idxs']})
        train_acc, val_acc = best_model['train_acc'], best_model['val_acc']
        print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
    with open(saved_model_path, 'wb') as f:
        pickle.dump(best_model, f)


def train_dp_shadow_model(exp, membership_idxs, train, val, test, seeds, device, save_dir, args, eval_train=True, target_model=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path_prefix = os.path.join(save_dir, f'exp_{exp}')
    saved_model_path = f'{save_path_prefix}_model.pickle'

    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
            best_model = pickle.load(f)
            if best_model['train_complete']:
                print(f'The model for experiment {exp+1} is already trained.')
                print(f'Number of epochs until the best model was found:', best_model['best_epoch'])
                train_acc, val_acc, test_acc = best_model['train_acc'], best_model['val_acc'], best_model['test_acc']
                print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
                return None

    seed_model, seed_batching, seed_dropout = seeds
    # Set the seed used to initialize the model.
    set_torch_seeds(seed_model)
    # Initialize the model using this seed.
    num_classes = get_num_classes(args.dataset)
    model = init_model(args.architecture, num_classes).to(device)
    
    errors = ModuleValidator.validate(model, strict=False)
    print('Errors DP Opacus', errors)

    # Align the model immediately after the initialization.
    if args.attacker_access == 'shadow_dataset_align_after_init':
        assert target_model is not None
        target_model = target_model.to(device)
        print('Aligning the model to the target model.')
        matching_method = HungarianAlgorithmMatching()
        model = TopDownWeightMatchingBasedAlignment(matching_method).\
                align_layers(model, target_model)

    # Reset the seed for dropout.
    set_torch_seeds(seed_dropout)

    # Set the seed for batching.
    g = torch.Generator()
    g.manual_seed(seed_batching)
    
    print('Training set size', len(train))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    seq_train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = init_optimizer(model, args.optimizer, args.learning_rate, args.momentum, args.weight_decay)
    learning_rate_scheduler = ExponentialLR(optimizer, gamma=0.5)

    criterion = nn.CrossEntropyLoss().to(device)

    privacy_engine = PrivacyEngine(secure_mode=True)
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.max_num_epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm
            )
    print(f"Using sigma={optimizer.noise_multiplier} and C={args.max_grad_norm}")

    logger = Logger(exp, args.print_every, save_path_prefix)
    best_val_acc = 0
    early_stopping_count = 0
    it = 1
    for epoch in range(args.max_num_epochs + 1):
        model.train()
        with BatchMemoryManager(data_loader=train_loader,
                max_physical_batch_size=128,
                optimizer=optimizer) as memory_safe_data_loader:
            train_iter = iter(memory_safe_data_loader)
            # Do not train for epoch 0, just evaluate the model.
            while True and epoch > 0:
                optimizer.zero_grad()
                try:
                    batch = next(train_iter)
                except StopIteration:
                    break

                inputs = batch[0].to(device)
                labels = batch[1].to(device)
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
        if eval_train:
            train_acc = evaluate_shadow_model(model, seq_train_loader, device)
            epoch_summary += f'train: {train_acc:.1%} '
        else:
            train_acc = 0
        print('Evaluating the model on the validation data.')
        val_acc = evaluate_shadow_model(model, val_loader, device)
        epsilon = privacy_engine.get_epsilon(args.delta)
        print(epoch_summary + f'validation: {val_acc:.1%}, epsilon: {epsilon:.2f}. Elapsed time: {time.time()-start_time:.2f} secs')
        logger.log_accuracy(train_acc, val_acc)

        # Save the model after each epoch (and the final model is the model from the last epoch).
        with open(saved_model_path, 'wb') as f:
            pickle.dump({'model_state_dict': model.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_complete': False,
                'best_epoch': epoch}, f)

    print('End of training. Loading the best model to mark the training as complete.')
    with open(saved_model_path, 'rb') as f:
        best_model = pickle.load(f)
        best_model['train_complete'] = True
        model = init_model(args.architecture, num_classes).to(device)
        model.load_state_dict(best_model['model_state_dict'])
        
        if 'best_epoch' in best_model:
            print('Best epoch: ', best_model['best_epoch'])
        test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc = evaluate_shadow_model(model, test_loader, device)
        best_model.update({'test_acc': test_acc, 
            'train_idxs': membership_idxs['train_idxs'], 
            'test_idxs': membership_idxs['test_idxs']})
        train_acc, val_acc = best_model['train_acc'], best_model['val_acc']
        print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
    with open(saved_model_path, 'wb') as f:
        pickle.dump(best_model, f)
