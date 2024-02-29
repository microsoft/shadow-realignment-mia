from collections import namedtuple

TrainArgs = namedtuple('TrainArgs', [
    'dataset',
    'batch_size', 
    'architecture', 
    'optimizer',
    'learning_rate',
    'momentum',
    'weight_decay',
    'print_every',
    'max_num_epochs',
    'num_epochs_patience',
    'min_learning_rate',
    'compute_auc',
    'num_workers'
    ])

MetaModelArgs = namedtuple('MetaModelArgs', [
    'model', # The model from which to extract features.
    'criterion', # Criterion use to compute the loss on the model (useful for the gradients).
    'features', # Which features to extract (activations/gradients).
    'layers', # From which layers to extract the features.
])
