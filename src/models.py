from collections import OrderedDict
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torchvision.models import vgg11, vgg13, vgg16, vgg19

from src.resnet import ResNet, resnet18


def init_optimizer(model, optimizer_name, learning_rate, momentum,
        weight_decay, nesterov=False):
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
            momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
            weight_decay=weight_decay)
    else:
        raise ValueError(f'ERROR: The {optimizer} optimizer is not currently supported.')
    return optimizer


def get_num_parameters(model):
    params = filter(lambda p: p.requires_grad, model.parameters())
    num_parameters = sum([np.prod(p.size()) for p in params])
    return num_parameters


def init_model(architecture, num_classes=10, verbose=True):
    if 'meta-model' in architecture:
        layer_component_architectures = architecture.split('__')[1:]
        #print('meta model', layer_component_architecture)
        model = MetaModel(layer_component_architectures, num_classes)
    elif 'set-based' in architecture:
        num_in_features = int(architecture.split('_')[1])
        model = SetBasedClassifier(num_in_features)
    elif 'generic-mlp-dropout' in architecture:
        layer_sizes = [int(n) for n in architecture.split('_')[1].split(',')]
        model = GenericMLPDropout(layer_sizes)
    elif 'generic-mlp' in architecture:
        #print('generic mlp', architecture)
        layer_sizes = [int(n) for n in architecture.split('_')[1].split(',')]
        model = GenericMLP(layer_sizes)
    elif 'cnn-gradients' in architecture:
        parameters = [int(n) for n in architecture.split('_')[1].split(',')]
        num_in_channels = parameters[0]
        num_gradients = parameters[1]
        num_out_channels = parameters[2]
        layer_sizes = parameters[2:]
        model = CNNGradients(num_in_channels, num_gradients, num_out_channels, layer_sizes)
    elif 'cnn-large-' in architecture:
        # Specify the architecture with a suffix to save the results in a 
        # different folder. This is for ablations and debugging purposes.
        model = CNNLarge(num_classes)
        #print(model)
    elif 'cnn-large' in architecture:
        model = CNNLarge(num_classes)
    elif 'vgg' in architecture:
        model = VGG(architecture, num_classes)
    elif 'resnet18' in architecture:
        model = resnet18(num_classes)
    else:
        raise ValueError(f'The architecture {architecture} is not currently supported.')

    if verbose:
        num_parameters = get_num_parameters(model)
        print(f'Initialized a model with the {architecture} architecture and {num_parameters} trainable parameters.')
    return model


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)

def compute_average_model(models):
    num_models = len(models)
    for i in range(1, num_models, 1):
        assert type(models[i]) == type(models[0])

    avg_model = copy.deepcopy(models[0]).eval()
    # Compute the mean in place according to:
    # x_0 = 0
    # x_{i+1}=i/(i+1)*x_i + 1/(i+1)*y_{i+1}
    # Hence x_n = 1/n * (y_1 + ... + y_n)
    if isinstance(models[0], CNNLarge):
        for i in range(1, num_models, 1):
            avg_model.conv1[0].weight = nn.Parameter(
                    avg_model.conv1[0].weight * i / (i + 1) + 
                    models[i].conv1[0].weight / (i + 1))
            avg_model.conv1[0].bias = nn.Parameter(
                    avg_model.conv1[0].bias * i / (i + 1) +
                    models[i].conv1[0].bias / (i + 1))
            
            avg_model.conv2[0].weight = nn.Parameter(
                    avg_model.conv2[0].weight * i / (i + 1) +
                    models[i].conv2[0].weight / (i + 1))
            avg_model.conv2[0].bias = nn.Parameter(
                    avg_model.conv2[0].bias * i / (i + 1) +
                    models[i].conv2[0].bias / (i + 1))

            avg_model.fc1.linear.weight = nn.Parameter(
                    avg_model.fc1.linear.weight * i / (i + 1)  +
                    models[i].fc1.linear.weight / (i + 1))
            avg_model.fc1.linear.bias = nn.Parameter(
                    avg_model.fc1.linear.bias * i / (i + 1) +
                    models[i].fc1.linear.bias / (i + 1))

            avg_model.fc2.linear.weight = nn.Parameter(
                    avg_model.fc2.linear.weight * i / (i + 1) +
                    models[i].fc2.linear.weight / (i + 1))
            avg_model.fc2.linear.bias = nn.Parameter(
                    avg_model.fc2.linear.bias * i / (i + 1) +
                    models[i].fc2.linear.bias / (i + 1))
    else:
        raise ValueError(f'ERROR: Unsupported model type {type(models[0])}')
        
    return avg_model


class CNNLarge(nn.Module):
    def __init__(self, num_classes, fc1_out_dim=500):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
            )
        self.flatten = Flatten()
        fc1_in_dim = 1250
        self.fc1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(fc1_in_dim, fc1_out_dim)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2))
        ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(fc1_out_dim, num_classes))
        ]))
        self.num_classes = num_classes


    def partial_forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x


    def forward(self, x):
        x = self.partial_forward(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, architecture, num_classes):
        super(VGG, self).__init__()

        # Initialize the convolutional part.
        in_channels = 3
        idx = 1
        self.conv_layer_names = []
        i = 0
        self.num_conv_layers = 0
        while i < len(cfg[architecture]):
            v = cfg[architecture][i]
            assert v != 'M'
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers = [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            if i < len(cfg[architecture])-1 and cfg[architecture][i+1] == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                i += 2
            else:
                i += 1
            setattr(self, f'conv{idx}', nn.Sequential(*layers))
            self.conv_layer_names.append(f'conv{idx}')
            self.num_conv_layers += 1
            idx += 1
            layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        # Initialize the classifier part.
        self.fc1 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(512, 512)),
            ('relu', nn.ReLU(True)),
            ('dropout', nn.Dropout())
            ]))
        self.fc2 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(512, 512)),
            ('relu', nn.ReLU(True))
            ]))
        self.fc3 = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(512, num_classes)),
            ]))

        self.num_classes = num_classes


    def partial_forward(self, x):
        for conv_layer_name in self.conv_layer_names:
            x = getattr(self, conv_layer_name)(x)
        x = x.view(x.size(0), -1)
        return x


    def forward(self, x):
        x = self.partial_forward(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


    def forward_per_layer(self, x):
        conv_outputs = []
        for conv_layer_name in self.conv_layer_names:
            x = getattr(self, conv_layer_name)(x)
            conv_outputs.append(x)
        fc_outputs = []
        x = self.fc1(x.view(x.size(0),-1))
        fc_outputs.append(x)
        x = self.fc2(x)
        fc_outputs.append(x)
        x = self.fc3(x)
        fc_outputs.append(x)
        return conv_outputs, fc_outputs


class CNNGradients(nn.Module):
    def __init__(self, num_in_channels, num_gradients, num_out_channels, layer_sizes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_in_channels, num_out_channels, kernel_size=(num_gradients, 1)),
            nn.Dropout(0.2)
        )
        self.mlp = GenericMLP(layer_sizes)
        self.output_size = self.mlp.output_size

    def forward(self, x):
        x = self.conv1(x)
        x = self.mlp(x)
        return x
        

@torch.no_grad()
def get_output_activations(model, x, labels, layers, architectures_only=False):
    model.eval()
    if isinstance(model, CNNLarge):
        c1 = model.conv1(x)
        c2 = model.conv2(c1)
        x1 = model.fc1(model.flatten(c2))
        x2 = F.softmax(model.fc2(x1), dim=1)
    elif isinstance(model, VGG):
        c = model.partial_forward(x)
        x1 = model.fc1(c)
        x2 = model.fc2(x1)
        x3 = F.softmax(model.fc3(x2), dim=1)
    elif isinstance(model, ResNet):
        x0 = model.partial_forward(x)
        x1 = F.softmax(model.fc1(x0), dim=1)
    elif isinstance(model, GenericMLP):
        x1 = model.fc1(x)
        if hasattr(model, 'fc2'):
            x2 = model.fc2(x1)
        else:
            x1 = F.softmax(x1, dim=1)
        if hasattr(model, 'fc3'):
            x3 = model.fc3(x2)
        elif hasattr(model, 'fc2'):
            x2 = F.softmax(x2, dim=1)
        if hasattr(model, 'fc4'):
            x4 = model.fc4(x3)
        elif hasattr(model, 'fc3'):
            x3 = F.softmax(x3, dim=1)
        if hasattr(model, 'fc5'):
            x5 = model.fc5(x4)
        elif hasattr(model, 'fc4'):
            x4 = F.softmax(x4, dim=1)
        if hasattr(model, 'fc6'):
            x6 = F.softmax(model.fc6(x5), dim=1)
        elif hasattr(model, 'fc5'):
            x5 = F.softmax(x5, dim=1)
    else:
        raise ValueError(f'ERROR: Invalid model architecture {type(model)}.')
    activations = []
    if architectures_only:
        architectures = []
    if 'conv1' in layers:
        activations.append(c1)
        if architectures_only:
            architectures.append(f'generic-mlp_{c1.numel()}')
    if 'conv2' in layers:
        activations.append(c2)
        if architectures_only:
            architectures.append(f'generic-mlp_{c2.numel()}')
    if 'conv5_x' in layers:
        assert isinstance(model, ResNet)
        c5 = model.conv5_x(
                model.conv4_x(
                    model.conv3_x(
                        model.conv2_x(
                            model.conv1(x)
                            )
                        )
                    )
                ).flatten(1)
        print(c5.size())
        activations.append(c5)
        if architectures_only:
            architectures.append(f'generic-mlp_{c5.numel()}')
    if 'fc1' in layers:
        if not 'fc1-ia-only' in layers:
            activations.append(x1)
            if architectures_only:
                architectures.append(f'generic-mlp_{x1.numel()}')
        if 'fc1-ia' in layers:
            weight = model.fc1.weight # Only for Resnet18.
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x0prim = x0.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x0prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if 'fc2' in layers:
        if not 'fc2-ia-only' in layers:
            activations.append(x2)
            #print(x2.max(), x2.min(), torch.histogram(x2, bins=20, range=(0,13)))
            if architectures_only:
                architectures.append(f'generic-mlp_{x2.numel()}')
        if 'fc2-ia' in layers:
            weight = model.fc2.linear.weight
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x1prim = x1.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x1prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if 'fc3' in layers:
        activations.append(x3)
        if architectures_only:
            architectures.append(f'generic-mlp_{x3.numel()}')
        if 'fc3-ia' in layers:
            weight = model.fc3.linear.weight
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x2prim = x2.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x2prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if 'fc4' in layers:
        activations.append(x4)
        if architectures_only:
            architectures.append(f'generic-mlp_{x4.numel()}')
        if 'fc4-ia' in layers:
            weight = model.fc4.linear.weight
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x3prim = x3.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x3prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if 'fc5' in layers:
        activations.append(x5)
        if architectures_only:
            architectures.append(f'generic-mlp_{x5.numel()}')
        if 'fc5-ia' in layers:
            weight = model.fc5.linear.weight
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x4prim = x4.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x4prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if 'fc6' in layers:
        activations.append(x6)
        if architectures_only:
            architectures.append(f'generic-mlp_{x6.numel()}')
        if 'fc6-ia' in layers:
            weight = model.fc6.linear.weight
            dprim, d = weight.size()
            batch_size = len(x)
            weight = weight.expand((batch_size, dprim, d))
            x5prim = x5.view(batch_size, 1, d).expand((batch_size, dprim, d))
            input_activations = x5prim * weight
            input_activations = torch.vstack([input_activations[i][labels[i]] for i in range(batch_size)])
            activations.append(input_activations.view(batch_size, -1))
            if architectures_only:
                assert batch_size == 1
                architectures.append(f'generic-mlp_{input_activations.numel()}')
    if len(activations) == 0:
        raise ValueError(f'ERROR: Invalid layer name {layers}.')
    if architectures_only:
        return architectures
    return activations


def get_gradients_conv_layer(conv_layer):
    gradients_w = conv_layer.weight.grad
    sizes = gradients_w.size()
    gradients_w = gradients_w.reshape(sizes[0], -1)
    gradients_b = conv_layer.bias.grad.reshape(sizes[0], 1)
    gradients_wb = torch.cat((gradients_w, gradients_b), dim=1)
    return gradients_wb.reshape(1, sizes[0], gradients_wb.size(1), 1)


def get_gradients_fc_layer(fc_layer):
    gradients_w = fc_layer.weight.grad
    dprim, d = gradients_w.size()
    gradients_b = fc_layer.bias.grad.reshape(dprim, 1)
    gradients_wb = torch.cat((gradients_w, gradients_b), dim=1)
    return gradients_wb.reshape(1, dprim, d+1, 1)


def get_gradients(model, criterion, x, label, layers, architectures_only=False):
    assert x.size(0) == 1, \
        'ERROR: Invalid sample size, set the --batch_size argument to 1.'
    model.eval()
    model.zero_grad()
    output = F.softmax(model(x), dim=1)
    loss = criterion(output, label)
    loss.backward()
    gradients = []
    if architectures_only:
        architectures = []
    if 'conv1' in layers:
        #gradients.append(torch.cat((model.conv1[0].weight.grad.flatten(), 
        #    model.conv1[0].bias.grad)).reshape(1, -1))
        #print(s1, s2)
        gradients.append(get_gradients_conv_layer(model.conv1[0]))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'conv2' in layers:
        #gradients.append(torch.cat((model.conv2[0].weight.grad.flatten(), 
        #    model.conv2[0].bias.grad)).reshape(1, -1))
        gradients.append(get_gradients_conv_layer(model.conv2[0]))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc1' in layers:
        #gradients.append(torch.cat((model.fc1.linear.weight.grad.flatten(), 
        #    model.fc1.linear.bias.grad)).reshape(1, -1))
        if isinstance(model, ResNet):
            gradients.append(get_gradients_fc_layer(model.fc1))
        else:
            gradients.append(get_gradients_fc_layer(model.fc1.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc2' in layers:
        #gradients.append(torch.cat((model.fc2.linear.weight.grad.flatten(), 
        #    model.fc2.linear.bias.grad)).reshape(1, -1))
        gradients.append(get_gradients_fc_layer(model.fc2.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc3' in layers:
        #gradients.append(torch.cat((model.fc3.linear.weight.grad.flatten(), 
        #    model.fc3.linear.bias.grad)).reshape(1, -1))
        gradients.append(get_gradients_fc_layer(model.fc3.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc4' in layers:
        gradients.append(get_gradients_fc_layer(model.fc4.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc5' in layers:
        gradients.append(get_gradients_fc_layer(model.fc5.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if 'fc6' in layers:
        gradients.append(get_gradients_fc_layer(model.fc6.linear))
        if architectures_only:
            _, num_in_channels, num_features, _ = gradients[-1].size()
            architectures.append(f'cnn-gradients_{num_in_channels},{num_features}')
    if len(gradients) == 0:
        raise ValueError(f'ERROR: Invalid layer name {layers}.')
    #gradients.append(label)
    if architectures_only:
        return architectures
    return gradients


def get_architectures(image, label, model, criterion, features, target_layers, 
        args, device):
    image, label, model, criterion = image.to(device), label.to(device), \
        model.to(device), criterion.to(device)
    if args.set_based:
        if hasattr(model, 'output_size'):
            num_classes = model.output_size
        elif hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        else:
            raise ValueError('ERROR: Unspecified number of classes.')
        return [f'generic-mlp_{num_classes},{args.meta_model_encoder_sizes}', 
                f'set-based_{args.num_set_based_features}']
    architectures = []
    if 'activations' in features:
        architectures.extend(get_output_activations(model, image, label, target_layers, architectures_only=True))
    if 'gradients' in features:
        architectures.extend(get_gradients(model, criterion, image, label, target_layers, architectures_only=True))
    for i in range(len(architectures)):
        if 'cnn-gradients' in architectures[i]:
            architectures[i] = f'{architectures[i]},{args.meta_model_kernel_size},{args.meta_model_encoder_sizes}'
        elif 'generic-mlp' in architectures[i]:
            architectures[i] = f'{architectures[i]},{args.meta_model_encoder_sizes}'
        else:
            raise ValueError(f'ERROR: Unsupported architecture {architectures[i]}')
    return architectures


def get_features(batch, model, criterion, features, target_layers, device,
        set_based=False):
    images = batch[0].to(device)
    labels = batch[1].to(device)
    mia_labels = batch[2].to(device)
    activations = []
    if 'activations' in features:
        activations = get_output_activations(model, images, labels, target_layers)
    gradients = []
    if 'gradients' in features:
        gradients.extend([[] for _ in range(len(target_layers.split(',')))])
        for i in range(len(images)):
            gradients_image = get_gradients(model, criterion, images[i:(i+1)], labels[i:(i+1)], target_layers)
            for j, layer_gradients in enumerate(gradients_image):
                gradients[j].append(layer_gradients)
        gradients = [torch.cat(layer_gradients, dim=0) for layer_gradients in gradients]
    # If `set_based` is set to True, then it will concatenate together the
    # output activation of a neuron, the input activation, and the gradients
    # into a single vector of features. This method assumes that only the 
    # features of the last layer are given.
    if set_based:
        oa_fc1 = activations[0].unsqueeze(2)
        oa_fc2 = activations[1]
        ia_fc2 = activations[2].unsqueeze(2)
        grad_fc2 = gradients[1].squeeze(3)[:, :, :-1].transpose(2, 1).\
                contiguous()
        grad_fc2_bias = gradients[0].squeeze(3)[:, :, -1:]
        #print(oa_fc2.size(), ia_fc2.size(), oa_fc1.size(), grad_fc2.size(),
        #        grad_fc2_bias.size())
        #print(gradients[0].size(), grad_fc2_bias.size())
        oa_ia_gradients = torch.cat([oa_fc1, ia_fc2, grad_fc2, grad_fc2_bias], 
                dim=2)
        #print(oa_ia_gradients.size())
        return [oa_fc2, oa_ia_gradients, labels], mia_labels
    else:
        return activations + gradients + [labels], mia_labels


class MetaModel(nn.Module):
    def __init__(self, layer_component_architectures, num_classes, embedding_dim=16):
        super().__init__()
        self.num_layers = len(layer_component_architectures)
        self.layer_components = nn.ModuleList([init_model(lca, num_classes) 
            for lca in layer_component_architectures])
        self.label_component = nn.Embedding(num_classes, embedding_dim)
        num_features = np.sum([lc.output_size for lc in self.layer_components]) + \
            self.label_component.embedding_dim
        self.classifier = GenericMLP((num_features, 128, 64, 2))

    
    def forward(self, x):
        x1 = torch.cat([self.layer_components[i](x[i]) 
            for i in range(self.num_layers)], dim=1)
        x2 = self.label_component(x[-1])
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)


class GenericMLP(nn.Module):
    def __init__(self, layer_sizes, flatten=True):
        super().__init__()
        if flatten:
            self.flatten = Flatten()
        #assert len(layer_sizes) > 2, 'ERROR: There should be at least one hidden layer.'
        self.layer_sizes = layer_sizes
        self.output_size = layer_sizes[-1]
        for i in range(len(layer_sizes) - 2):
            layer = nn.Sequential(OrderedDict([
                ('linear', nn.Linear(layer_sizes[i], layer_sizes[i+1])),
                ('relu', nn.ReLU())
                ]))
            setattr(self, f'fc{i+1}', layer)
        last_layer = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        ]))
        setattr(self, f'fc{len(layer_sizes)-1}', last_layer)


    def partial_forward(self, x):
        if hasattr(self, 'flatten'):
            return self.flatten(x)
        return x


    def forward(self, x):
        x = self.partial_forward(x)
        for i in range(len(self.layer_sizes)-1):
            x = getattr(self, f'fc{i+1}')(x)
        return x


class GenericMLPDropout(GenericMLP):
    def __init__(self, layer_sizes):
        super().__init__(layer_sizes)
        #assert len(layer_sizes) > 2, 'ERROR: There should be at least one hidden layer.'
        for i in range(len(self.layer_sizes) - 2):
            layer_i = getattr(self, f'fc{i+1}')
            layer_i.append(nn.Dropout(0.05))


class SetBasedClassifier(nn.Module):
    def __init__(self, num_in_features):
        super().__init__()
        self.phi = GenericMLP((num_in_features, 128, 64), flatten=False)
        self.output_size = 64


    def forward(self, x):
        # x is of dimension batch size x num neurons x num features 
        # (activations and gradients).
        #print('x size', x.size())
        embeddings = self.phi(x).sum(axis=1)
        return embeddings


@torch.no_grad()
def map_to_canonical(model, fc_layer_names):
    canonical_model = copy.deepcopy(model)
    if isinstance(model, CNNTutorial):
        conv_layer1 = canonical_model.conv1[0]
        #print('Convlayer 1', conv_layer1.weight.size())
        out_channels1 = conv_layer1.weight.size(0)
        sort_idxs1 = torch.argsort(torch.sum(conv_layer1.weight.view(out_channels1, -1), dim=1))

        conv_layer1.weight = Parameter(conv_layer1.weight[sort_idxs1])
        conv_layer1.bias = Parameter(conv_layer1.bias[sort_idxs1])

        #print('Convlayer 1 after', conv_layer1.weight.size())

        conv_layer2 = canonical_model.conv2[0]
        #print('Convlayer 2', conv_layer2.weight.size())

        conv_layer2.weight = Parameter(conv_layer2.weight[:, sort_idxs1, :, :])
        #print('Convlayer 2 after 1', conv_layer2.weight.size())

        
        out_channels2 = conv_layer2.weight.size(0)
        sort_idxs2 = torch.argsort(torch.sum(conv_layer2.weight.view(out_channels2, -1), dim=1))

        conv_layer2.weight = Parameter(conv_layer2.weight[sort_idxs2])
        conv_layer2.bias = Parameter(conv_layer2.bias[sort_idxs2])

        #print('Convlayer 2 after 2', conv_layer2.weight.size())

        next_layer = canonical_model.fc1.linear
        #print('Fc1', next_layer.weight.size())
        K1, K2 = conv_layer2.weight.size(2), conv_layer2.weight.size(3)
        Dprim, D = next_layer.weight.size()
        assert D == K1 * K2 * out_channels2
        W = next_layer.weight.view(Dprim, out_channels2, K1, K2)
        next_layer.weight = Parameter(W[:, sort_idxs2, :, :].view(Dprim, D))
        #print('Fc1 after', next_layer.weight.size())


    for li in range(len(fc_layer_names) - 1):
        layer_name = fc_layer_names[li]
        assert hasattr(canonical_model, layer_name)
        layer = getattr(canonical_model, layer_name)
        sort_idxs = torch.argsort(torch.sum(layer.linear.weight, dim=1))
        layer.linear.weight = Parameter(layer.linear.weight[sort_idxs])
        layer.linear.bias = Parameter(layer.linear.bias[sort_idxs])
        #if li < len(fc_layer_names) - 1:
        next_layer = getattr(canonical_model, fc_layer_names[li+1])
        #print(li, next_layer.linear.weight.size(), sort_idxs)
        next_layer.linear.weight = Parameter(next_layer.linear.weight[:, sort_idxs])
    return canonical_model



if __name__ == '__main__':
    model = VGG('vgg16', num_classes=10)
    print(model)
    #print('conv10', model.conv10[0].weight.size())
    #print('fc1', model.fc1.linear.weight.size())
    #print('fc2', model.fc2.linear.weight.size())
    sample = torch.normal(1, 1, size=(1, 3, 32, 32))
    conv_layers, fc_layers = model.forward_per_layer(sample)
    for layers in [conv_layers, fc_layers]:
        for l in layers:
            print(l.size())
