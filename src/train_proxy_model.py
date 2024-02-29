import os
import pickle
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.dataset import get_num_classes
from src.logger import Logger
from src.models import init_model, init_optimizer, CNNLarge


def evaluate_proxy_model(model, dataloader, device):
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


def get_layer_names(model):
    if isinstance(model, CNNLarge):
        return ['conv1', 'conv2', 'fc1', 'fc2']
    else:
        raise TypeError(f'ERROR: Invalid model type {type(model)}')


def top(model: torch.nn.Module, start_layer: int=0, reset_weights=False):
    layer_names = get_layer_names(model)
    for layer_name in layer_names[:start_layer]:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = False
    for layer_name in layer_names[start_layer:]:
        layer = getattr(model, layer_name)
        for param in layer.parameters():
            param.requires_grad = True

        if reset_weights:
            assert hasattr(layer[0], 'reset_parameters')
            print(f'Reinitializing the weights of layer {layer_name}')
            layer[0].reset_parameters()
    return model


def train_proxy_model(exp, target_model, start_layer, membership_idxs, 
        train, val, test, seeds, device, save_dir, args, eval_train=True):

    num_classes = get_num_classes(args.dataset)
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
                
                model = init_model(args.architecture, num_classes).to(device)
                model.load_state_dict(best_model['model_state_dict'])
                return model

    seed_model, seed_batching, seed_dropout = seeds
    # Set the seed used to initialize the model.
    set_torch_seeds(seed_model)
    # Initialize the model using this seed.
    model = top(target_model, start_layer, reset_weights=True).to(device)

    # Reset the seed for dropout.
    set_torch_seeds(seed_dropout)

    # Set the seed for batching.
    g = torch.Generator()
    g.manual_seed(seed_batching)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)

    seq_train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = init_optimizer(model, args.optimizer, args.learning_rate, args.momentum, args.weight_decay)
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
            #print(model.conv1[0].bias[:2], print(model.fc1[0].bias[:2]))
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
            train_acc = evaluate_proxy_model(model, seq_train_loader, device)
            epoch_summary += f'train: {train_acc:.1%} '
        else:
            train_acc = 0
        print('Evaluating the model on the validation data.')
        val_acc = evaluate_proxy_model(model, val_loader, device)
        print(epoch_summary + f'validation: {val_acc:.1%}. Elapsed time: {time.time()-start_time:.2f} secs')
        logger.log_accuracy(train_acc, val_acc)

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
            if epoch > 0 and epoch % 10 == 0:
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
        test_acc = evaluate_proxy_model(model, test_loader, device)
        best_model.update({'test_acc': test_acc, 
            'train_idxs': membership_idxs['train_idxs'], 
            'test_idxs': membership_idxs['test_idxs']})
        train_acc, val_acc = best_model['train_acc'], best_model['val_acc']
        print(f'Train acc: {train_acc:.1%}, val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
    with open(saved_model_path, 'wb') as f:
        pickle.dump(best_model, f)

    return model

