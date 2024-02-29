import math
import numpy as np
import os
import pickle
import random
from sklearn.metrics import accuracy_score, auc, roc_curve
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from src.dataset import get_num_classes
from src.logger import Logger
from src.models import get_features, init_model, init_optimizer


def evaluate_meta_model(model, dataloader, best_threshold=None):
    model.eval()
    correct = 0
    total = 0
    y_true, y_pred = [], []
    for batch in tqdm(dataloader):
        inputs, labels = batch
        with torch.no_grad():
            outputs = model(inputs)
        softmax = F.softmax(outputs, dim=1)
        y_pred.extend(softmax[:, 1].tolist())
        y_true.extend(labels.cpu().tolist())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if best_threshold is None:
        #print(len(fpr), len(tpr), fpr[0], tpr[0])
        #print(fpr, tpr)
        acc_thresholds = 1-(fpr+(1-tpr))/2
        #print(acc_thresholds)
        best_acc = np.max(acc_thresholds)
        best_threshold = thresholds[np.argmax(acc_thresholds)]
    else:
        # What are the FPR and TPR values for threshold?
        y_pred_t = y_pred >= best_threshold
        best_acc = accuracy_score(y_true, y_pred_t)
    auc_score = auc(fpr, tpr)
    return {'acc': acc, 'fpr': fpr, 'tpr': tpr, 'auc': auc_score, \
        'best_acc': best_acc, 'best_threshold': best_threshold}


class MetaModelDataLoader(object):
    def __init__(self, dataset, models, mia_labels, batch_size, shuffle, 
            device, criterion, features, target_layers, set_based):
        if isinstance(dataset[0], torch.utils.data.Dataset):
            print("Detecting multiple datasets (typically for VGG + imbalanced train/test split)")
            # One dataset for every model (unbalanced scenario).
            self.images = [torch.cat([image.unsqueeze(0) for image, _ in d], dim=0)
                    for d in dataset]
            self.labels = [torch.LongTensor([label for _, label in d])
                    for d in dataset]
            self.num_records = len(self.images[0])
            for i in range(len(self.images)):
                assert self.num_records == len(self.images[i]) == len(self.labels[i])
            self.multiple_datasets = True
        else:
            # One common dataset for every model (balanced scenario).
            self.images = torch.cat([image.unsqueeze(0) for image, _ in dataset], dim=0)
            self.labels = torch.LongTensor([label for _, label in dataset])
            self.num_records = len(self.images)
            assert self.num_records == len(self.labels)
            self.multiple_datasets = False
        assert len(models) == len(mia_labels), \
            f'ERROR: The number of membership label lists {len(mia_labels)} does not match the number of models {len(models)}.'
        self.models = [model.to(device) for model in models]
        self.mia_labels = []
        for i in range(len(mia_labels)):
            assert len(mia_labels[i]) == self.num_records, \
                f'ERROR: The number of membership labels {len(mia_labels[i])} does not match the number of records {len(self.images)}.'
            self.mia_labels.append(torch.LongTensor(mia_labels[i]))
        self.batch_size = batch_size
        self.num_batches = self.num_records // self.batch_size
        if self.num_records % self.batch_size > 0:
            self.num_batches += 1
        self.shuffle = shuffle
        self.device = device
        self.criterion = criterion.to(device)
        self.features = features
        self.target_layers = target_layers
        self.set_based = set_based


    def __iter__(self):
        if self.shuffle:
            self.record_idxs = np.random.permutation(self.num_records)
            self.model_idxs = []
            while len(self.model_idxs) < self.num_batches:
                self.model_idxs.extend(list(np.random.permutation(len(self.models))))
            self.model_idxs = self.model_idxs[:self.num_batches]
        else:
            self.record_idxs = np.arange(self.num_records)
            self.model_idxs = []
            while len(self.model_idxs) < self.num_batches:
                self.model_idxs.extend(list(range(len(self.models))))
            self.model_idxs = self.model_idxs[:self.num_batches]
        self.curr_batch_start_idx = 0
        self.curr_model_idx = 0
        return self


    def __next__(self):
        if self.curr_batch_start_idx >= self.num_records:
            assert self.curr_model_idx == self.num_batches, \
                f'ERROR: Has not reached the last batch {self.curr_model_idx}'
            self.curr_batch_start_idx = 0
            self.curr_model_idx = 0
            raise StopIteration()

        curr_batch_end_idx = min(self.curr_batch_start_idx + self.batch_size, self.num_records)
        batch_idxs = self.record_idxs[self.curr_batch_start_idx:curr_batch_end_idx]
        if self.multiple_datasets is True:
            images = self.images[self.model_idxs[self.curr_model_idx]][batch_idxs].to(self.device)
            labels = self.labels[self.model_idxs[self.curr_model_idx]][batch_idxs].to(self.device)
        else:
            images = self.images[batch_idxs].to(self.device)
            labels = self.labels[batch_idxs].to(self.device)
        mia_labels = self.mia_labels[self.model_idxs[self.curr_model_idx]][batch_idxs].to(self.device) 
        batch = (images, labels, mia_labels)

        model = self.models[self.model_idxs[self.curr_model_idx]]

        features, labels = get_features(batch, model, self.criterion, 
                self.features, self.target_layers, self.device, self.set_based)

        features = [f.to(self.device) for f in features]
        labels = labels.to(self.device)

        self.curr_batch_start_idx = curr_batch_end_idx
        self.curr_model_idx += 1

        return features, labels


def print_metrics(model):
    val_acc, test_acc = model['val_metrics']['acc'], model['test_metrics']['acc']
    print(f'Val acc: {val_acc:.1%}, test acc: {test_acc:.1%}')
    val_auc, test_auc = model['val_metrics']['auc'], model['test_metrics']['auc']
    print(f'Val auc: {val_auc:.3f}, test auc: {test_auc:.3f}')
    val_best_acc, test_best_acc = model['val_metrics']['best_acc'], model['test_metrics']['best_acc']
    print(f'Val best acc: {val_best_acc:.1%}, test best acc: {test_best_acc:.1%}')
    return model


def train_meta_model(exp, dataset, models, mia_labels, device, save_dir, args, 
        meta_model_architecture, eval_train=False):
    save_path_prefix = os.path.join(save_dir, f'exp_{exp}')
    saved_model_path = f'{save_path_prefix}_model.pickle'

    if os.path.exists(saved_model_path):
        with open(saved_model_path, 'rb') as f:
            best_model = pickle.load(f)
            if best_model['train_complete']:
                print(f'The model for experiment {exp+1} is already trained.')
                print_metrics(best_model)
                return None

    # Set the seed of the random number generator. Note that we use the
    # experiment seed, rather than the global seed (`args.seed`).
    torch.manual_seed(args.meta_model_seed)
    np.random.seed(args.meta_model_seed)
    random.seed(args.meta_model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.meta_model_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader = MetaModelDataLoader(dataset['train'], 
        models['train'], 
        mia_labels['train'], 
        batch_size=args.meta_model_batch_size, 
        shuffle=True, 
        device=device,
        criterion=criterion,
        features=args.target_model_features,
        target_layers=args.target_model_layers,
        set_based=args.set_based)
    seq_train_loader = MetaModelDataLoader(dataset['train'], 
        models['train'], 
        mia_labels['train'], 
        batch_size=args.meta_model_batch_size, 
        shuffle=False, 
        device=device,
        criterion=criterion,
        features=args.target_model_features,
        target_layers=args.target_model_layers,
        set_based=args.set_based)
    val_loader = MetaModelDataLoader(dataset['val'], 
        models['val'], 
        mia_labels['val'], 
        batch_size=args.meta_model_batch_size, 
        shuffle=True, 
        device=device,
        criterion=criterion,
        features=args.target_model_features,
        target_layers=args.target_model_layers,
        set_based=args.set_based)

    num_classes = get_num_classes(args.dataset)
    meta_model = init_model(meta_model_architecture, num_classes).to(device)
    print(meta_model)

    optimizer = init_optimizer(meta_model, args.meta_model_optimizer, args.meta_model_learning_rate, 
        args.meta_model_momentum, args.meta_model_weight_decay)
    learning_rate_scheduler = ExponentialLR(optimizer, gamma=0.5)

    logger = Logger(exp, args.print_every, save_path_prefix)
    best_val_acc = 0
    early_stopping_count = 0
    it = 1
    for epoch in range(args.meta_model_max_num_epochs + 1):
        meta_model.train()
        train_iter = iter(train_loader)
        # Do not train for epoch 0, just evaluate the model.
        while True and epoch > 0:
            optimizer.zero_grad()
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                break
            outputs = meta_model(inputs)
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
            train_metrics = evaluate_meta_model(meta_model, seq_train_loader)
            train_acc = train_metrics['acc']
            epoch_summary += f'train: {train_acc:.1%} '
        else:
            train_metrics, train_acc = None, 0
        print('Evaluating the model on the validation data.')
        val_metrics = evaluate_meta_model(meta_model, val_loader)
        val_acc = val_metrics['acc']
        print(epoch_summary + f'validation: {val_acc:.1%}. Elapsed time: {time.time()-start_time:.2f} secs')
        logger.log_accuracy(train_acc, val_acc)

        if val_acc > best_val_acc:
            print(f'The validation accuracy has improved from {best_val_acc:.1%} to {val_acc:.1%}. Saving the parameters to disk.')
            best_val_acc = val_acc
            
            with open(saved_model_path, 'wb') as f:
                pickle.dump({'model_state_dict': meta_model.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics' : val_metrics,
                    'train_complete': False}, f)
        else:
            early_stopping_count += 1
            if early_stopping_count == args.meta_model_num_epochs_patience:
                early_stopping_count = 0
                learning_rate_scheduler.step()
                new_learning_rate = optimizer.param_groups[0]['lr']
                print(f'New learning rate: {new_learning_rate}')
                if new_learning_rate < args.meta_model_min_learning_rate:
                    print(f'Stopping the training because the learning rate is lower than {args.meta_model_min_learning_rate}.')
                    break

    print('End of training. Loading the best model to mark the training as complete.')
    with open(saved_model_path, 'rb') as f:
        best_model = pickle.load(f)
        best_model['train_complete'] = True
        meta_model = init_model(meta_model_architecture, num_classes).to(device)
        meta_model.load_state_dict(best_model['model_state_dict'])

        best_threshold = best_model['val_metrics']['best_threshold']
        print(f'Best threshold as determined from the validation set: {best_threshold:.2f}')
        
        test_loader = MetaModelDataLoader(dataset['test'], 
            models['test'], 
            mia_labels['test'], 
            batch_size=args.meta_model_batch_size, 
            shuffle=True, 
            device=device,
            criterion=criterion,
            features=args.target_model_features,
            target_layers=args.target_model_layers,
            set_based=args.set_based)
        test_metrics = evaluate_meta_model(meta_model, test_loader, best_threshold)
        best_model.update({'test_metrics': test_metrics})
        if eval_train:
            train_acc = best_model['train_metrics']['acc']
            print(f'Train acc: {train_acc:.1%}')
        print_metrics(best_model)
    del best_model['model_state_dict']
    with open(saved_model_path, 'wb') as f:
        pickle.dump(best_model, f)
