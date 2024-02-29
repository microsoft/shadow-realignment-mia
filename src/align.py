import copy
from scipy.optimize import linear_sum_assignment
import numpy as np
import time
import torch
from torch.nn import Parameter


from src.models import CNNLarge, GenericMLP, VGG
from src.resnet import ResNet


class AlignmentMethod(object):
    """
    The current class supports architectures with two convolutional layers and two 
    fully connected layers.
    """

    @staticmethod
    def check_layers(model):
        if isinstance(model, CNNLarge):
            assert hasattr(model, 'conv1')
            assert hasattr(model, 'conv2')
            assert hasattr(model, 'fc1')
            assert hasattr(model, 'fc2')
        elif isinstance(model, GenericMLP):
            num_layers = len(model.layer_sizes) - 1
            for i in range(num_layers):
                assert hasattr(model, f'fc{i+1}')
        elif isinstance(model, VGG):
            assert hasattr(model, 'fc3') and hasattr(model, 'fc2') and hasattr(model, 'fc1')
        elif isinstance(model, ResNet):
            assert hasattr(model, 'fc1')
        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}')


    def align_layers(self, model, ref_model=None, sorted_idxs=None, records=None):
        self.check_layers(model)
        if ref_model is not None:
            self.check_layers(ref_model)
            if records is None:
                #print('I am here')
                return self.sort_layers(model, ref_model)
            else:
                return self.sort_layers(model, ref_model, records)
        elif sorted_idxs is not None:
            return self.sort_layers(model, sorted_idxs)
        else:
            # Identity or any other alignment based solely on the model.
            return self.sort_layers(model)


    def sort_layers(self):

        raise NotImplementedError


class BottomUpAlignmentMethod(AlignmentMethod):

    def align_conv_layer(self, model, sorted_idxs, conv_idx=1):
        # Sorting the first convolutional layer.
        conv = f'conv{conv_idx}'
        #print(model)
        getattr(model, conv)[0].weight = Parameter(getattr(model, conv)[0].weight[sorted_idxs])
        getattr(model, conv)[0].bias = Parameter(getattr(model, conv)[0].bias[sorted_idxs])

        # Propagating the permutation to the next layer.
        next_conv = f'conv{conv_idx+1}'
        getattr(model, next_conv)[0].weight = Parameter(getattr(model, next_conv)[0].\
                weight[:, sorted_idxs, :, :])
        return model


    def align_junction_layer(self, model, sorted_idxs, conv_idx=2):
        conv = f'conv{conv_idx}'
        # Sorting the second convolutional layer.
        getattr(model, conv)[0].weight = Parameter(getattr(model, conv)[0].weight[sorted_idxs])
        getattr(model, conv)[0].bias = Parameter(getattr(model, conv)[0].bias[sorted_idxs])

        # Propagating the permutation to the next layer.
        C_out, _, _, _ = getattr(model, conv)[0].weight.size()
        #print(model)
        #print(conv, getattr(model, conv)[0].weight.size(), model.fc1.linear.weight.size())
        Dprim, D = model.fc1.linear.weight.size()
        #assert D == K1 * K2 * C_out
        W = model.fc1.linear.weight.view(Dprim, C_out, -1)
        model.fc1.linear.weight = Parameter(W[:, sorted_idxs, :].view(Dprim, D))
        return model


    def align_fc1_layer(self, model, sorted_idxs):

        # Sorting the first fully connected layer.
        model.fc1.linear.weight = Parameter(model.fc1.linear.weight[sorted_idxs])
        model.fc1.linear.bias = Parameter(model.fc1.linear.bias[sorted_idxs])

        # Propagating the permutation to the next layer.
        model.fc2.linear.weight = Parameter(model.fc2.linear.weight[:, sorted_idxs])
        return model


    def align_fc_layer(self, model, fc_idx, sorted_idxs):
        current_layer = getattr(model, f'fc{fc_idx}')
        next_layer = getattr(model, f'fc{fc_idx+1}')

        # Sorting the first fully connected layer.
        current_layer.linear.weight = Parameter(current_layer.linear.weight[sorted_idxs])
        current_layer.linear.bias = Parameter(current_layer.linear.bias[sorted_idxs])

        # Propagating the permutation to the next layer.
        next_layer.linear.weight = Parameter(next_layer.linear.weight[:, sorted_idxs])

        return model


    def sort_layers(self, model, sorted_idxs):
        aligned_model = copy.deepcopy(model)
        if isinstance(model, CNNLarge):
            assert sorted(list(sorted_idxs.keys())) == ['conv1', 'conv2', 'fc1']

            aligned_model = self.align_conv_layer(aligned_model, sorted_idxs['conv1'], conv_idx=1)
            aligned_model = self.align_junction_layer(aligned_model, sorted_idxs['conv2'])
            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs['fc1'])
        elif isinstance(model, GenericMLP):
            num_layers = len(aligned_model.layer_sizes) - 1
            assert sorted(list(sorted_idxs.keys())) == \
                [f'fc{i+1}' for i in range(num_layers-1)]
            for i in range(num_layers - 1):
                fc_idx = i + 1
                aligned_model = self.align_fc_layer(model, fc_idx, sorted_idxs[f'fc{fc_idx}'])
        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}.')
        
        return aligned_model


class TopDownAlignmentMethod(AlignmentMethod):

    def align_fc2_layer(self, model, sorted_idxs):
        ## Alignment of the last layer.
        model.fc2.linear.weight = Parameter(model.fc2.linear.weight[:, sorted_idxs])
        # Propagating the permutation to the layer below.
        model.fc1.linear.weight = Parameter(model.fc1.linear.weight[sorted_idxs])
        model.fc1.linear.bias = Parameter(model.fc1.linear.bias[sorted_idxs])

        return model


    def align_fc1_layer(self, model, sorted_idxs):
        ## Alignment of the second to last layer.
        C_out, _, K1, K2 = model.conv2[0].weight.size()
        Dprim, D = model.fc1.linear.weight.size()
        assert D == K1 * K2 * C_out
        W = model.fc1.linear.weight.view(Dprim, C_out, K1, K2)
        model.fc1.linear.weight = Parameter(W[:, sorted_idxs, :, :].view(Dprim, D))
        # Propagating the permutation to the next layer.
        model.conv2[0].weight = Parameter(model.conv2[0].weight[sorted_idxs])
        model.conv2[0].bias = Parameter(model.conv2[0].bias[sorted_idxs])

        return model

    def align_conv2_layer(self, model, sorted_idxs):
        ## Alignment of the third to last layer.
        model.conv2[0].weight = Parameter(model.conv2[0].weight[:, sorted_idxs, :, :])
        # Propagating the permutation to the next layer.
        model.conv1[0].weight = Parameter(model.conv1[0].weight[sorted_idxs])
        model.conv1[0].bias = Parameter(model.conv1[0].bias[sorted_idxs])
        
        return model


    def align_fc_layer(self, model, fc_idx, sorted_idxs):
        current_layer = getattr(model, f'fc{fc_idx}')
        next_layer = getattr(model, f'fc{fc_idx - 1}')
        assert len(sorted_idxs) == current_layer.linear.weight.size(1) == next_layer.linear.weight.size(0)
        ## Top-down alignment of a fully-connected layer.
        current_layer.linear.weight = Parameter(current_layer.linear.weight[:, sorted_idxs])
        # Propagating the permutation to the layer below.
        next_layer.linear.weight = Parameter(next_layer.linear.weight[sorted_idxs])
        next_layer.linear.bias = Parameter(next_layer.linear.bias[sorted_idxs])

        return model


    def sort_layers(self, model, sorted_idxs):
        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            assert sorted(list(sorted_idxs.keys())) == ['conv2', 'fc1', 'fc2']
            
            aligned_model = self.align_fc2_layer(aligned_model, sorted_idxs['fc2'])
            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs['fc1'])
            aligned_model = self.align_conv2_layer(aligned_model, sorted_idxs['conv2'])
        elif isinstance(model, GenericMLP):            
            num_layers = len(aligned_model.layer_sizes) - 1
            
            assert sorted(list(sorted_idxs.keys())) == \
                [f'fc{num_layers - i}' for i in range(num_layers-1)][::-1]
            
            for i in range(num_layers - 1):
                fc_idx = num_layers - i
                aligned_model = self.align_fc_layer(aligned_model, fc_idx, sorted_idxs[f'fc{fc_idx}'])
        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}.')

        return aligned_model


class WeightSortingBasedAlignment(BottomUpAlignmentMethod):

    def sort_layers(self, model):
        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            D1 = aligned_model.conv1[0].weight.size(0)
            sorted_idxs_conv1 = torch.argsort(torch.sum(aligned_model.conv1[0].weight.view(D1, -1), dim=1))
            aligned_model = self.align_conv_layer(aligned_model, sorted_idxs_conv1, conv_idx=1)

            D2 = aligned_model.conv2[0].weight.size(0)
            sorted_idxs_conv2 = torch.argsort(torch.sum(aligned_model.conv2[0].weight.view(D2, -1), dim=1))
            aligned_model = self.align_junction_layer(aligned_model, sorted_idxs_conv2)

            sorted_idxs_fc1 = torch.argsort(torch.sum(aligned_model.fc1.linear.weight, dim=1))
            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)

        elif isinstance(model, GenericMLP):
            num_layers = len(aligned_model.layer_sizes) - 1
            for i in range(num_layers - 1):
                fc_idx = i + 1
                sorted_idxs_fc = torch.argsort(\
                    torch.sum( getattr(aligned_model, f'fc{fc_idx}').linear.weight, dim=1) )
                aligned_model = self.align_fc_layer(aligned_model, fc_idx, sorted_idxs_fc)
        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}.')
        
        return aligned_model


class BottomUpWeightMatchingBasedAlignment(BottomUpAlignmentMethod):

    def __init__(self, matching_method):
        super().__init__()
        self.matching_method = matching_method


    @torch.no_grad()
    def sort_layers(self, model, ref_model):
        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            self.elapsed_times = []
            # Sort the first convolutional layer.
            D1 = aligned_model.conv1[0].weight.size(0)
            assert D1 == ref_model.conv1[0].weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `conv1` output channels.'
            conv1_w = torch.cat( (aligned_model.conv1[0].weight.view(D1, -1), aligned_model.conv1[0].bias.view(D1, 1)), dim=1)
            conv1_w_ref = torch.cat( (ref_model.conv1[0].weight.view(D1, -1), ref_model.conv1[0].bias.view(D1, 1) ), dim=1)
            conv1_distances = torch.cdist(conv1_w_ref, conv1_w, p=2)
            start_time = time.time()
            sorted_idxs_conv1 = self.matching_method.match(conv1_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_conv_layer(aligned_model, sorted_idxs_conv1)

            # Sort the second convolutional layer.
            D2 = aligned_model.conv2[0].weight.size(0)
            assert D2 == ref_model.conv2[0].weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `conv2` output channels.'
            conv2_w = torch.cat( (aligned_model.conv2[0].weight.view(D2, -1), aligned_model.conv2[0].bias.view(D2, 1)), dim=1)
            conv2_w_ref = torch.cat( (ref_model.conv2[0].weight.view(D2, -1), ref_model.conv2[0].bias.view(D2, 1) ), dim=1)
            conv2_distances = torch.cdist(conv2_w_ref, conv2_w, p=2)
            start_time = time.time()
            sorted_idxs_conv2 = self.matching_method.match(conv2_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_junction_layer(aligned_model, sorted_idxs_conv2)

            # Sort the first fully connected layer.
            D3 = aligned_model.fc1.linear.weight.size(0)
            assert D3 == ref_model.fc1.linear.weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
            fc1_w = torch.cat( (aligned_model.fc1.linear.weight.view(D3, -1), aligned_model.fc1.linear.bias.view(D3, 1) ), dim=1)
            fc1_w_ref = torch.cat( (ref_model.fc1.linear.weight.view(D3, -1), ref_model.fc1.linear.bias.view(D3, 1) ), dim=1)
            fc1_distances = torch.cdist(fc1_w_ref, fc1_w, p=2)
            start_time = time.time()
            sorted_idxs_fc1 = self.matching_method.match(fc1_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)
        elif isinstance(model, GenericMLP):
            for i in range(len(aligned_model.layer_sizes) - 2):
                fc_idx = i + 1
                fc_name = f'fc{fc_idx}'
                # Sort the fully connected layer.
                aligned_model_fc = getattr(aligned_model, fc_name)
                ref_model_fc = getattr(ref_model, fc_name)
                D = aligned_model_fc.linear.weight.size(0)
                assert D == ref_model_fc.linear.weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
                fc_w = torch.cat( (aligned_model_fc.linear.weight.view(D, -1), aligned_model_fc.linear.bias.view(D, 1) ), dim=1)
                fc_w_ref = torch.cat( (ref_model_fc.linear.weight.view(D, -1), ref_model_fc.linear.bias.view(D, 1) ), dim=1)
                fc_distances = torch.cdist(fc_w_ref, fc_w, p=2)
                sorted_idxs = self.matching_method.match(fc_distances)
                aligned_model = self.align_fc_layer(aligned_model, fc_idx, sorted_idxs)
        elif isinstance(model, VGG):
            for i in range(model.num_conv_layers):
                conv = f'conv{i+1}'
                D = getattr(aligned_model, conv)[0].weight.size(0)
                assert D == getattr(ref_model, conv)[0].weight.size(0), \
                        f'ERROR: `aligned_model` and `ref_model` have different number of `conv{i+1}` output channels.'
                conv_w = torch.cat( (getattr(aligned_model, conv)[0].weight.view(D, -1), 
                    getattr(aligned_model, conv)[0].bias.view(D, 1)), dim=1)
                conv_w_ref = torch.cat( (getattr(ref_model, conv)[0].weight.view(D, -1), 
                            getattr(ref_model, conv)[0].bias.view(D, 1) ), dim=1)  
                conv_distances = torch.cdist(conv_w_ref, conv_w, p=2)
                sorted_idxs_conv = self.matching_method.match(conv_distances)

                if i < model.num_conv_layers - 1:
                    aligned_model = self.align_conv_layer(aligned_model, sorted_idxs_conv, i+1)
                else:
                    aligned_model = self.align_junction_layer(aligned_model, sorted_idxs_conv, 
                            model.num_conv_layers)

            for i in [1, 2]:
                fc_name = f'fc{i}'
                # Sort the fully connected layer.
                aligned_model_fc = getattr(aligned_model, fc_name)
                ref_model_fc = getattr(ref_model, fc_name)
                D = aligned_model_fc.linear.weight.size(0)
                assert D == ref_model_fc.linear.weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
                fc_w = torch.cat( (aligned_model_fc.linear.weight.view(D, -1), aligned_model_fc.linear.bias.view(D, 1) ), dim=1) 
                fc_w_ref = torch.cat( (ref_model_fc.linear.weight.view(D, -1), ref_model_fc.linear.bias.view(D, 1) ), dim=1)
                fc_distances = torch.cdist(fc_w_ref, fc_w, p=2)
                sorted_idxs = self.matching_method.match(fc_distances)
                aligned_model = self.align_fc_layer(aligned_model, i, sorted_idxs)

        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}')

        return aligned_model


class TopDownWeightMatchingBasedAlignment(TopDownAlignmentMethod):

    def __init__(self, matching_method):
        super().__init__()
        self.matching_method = matching_method


    @torch.no_grad()
    def sort_layers(self, model, ref_model):
        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):

            # Sort the first (from the top) fully connected layer.
            D1 = aligned_model.fc2.linear.weight.size(1)
            assert D1 == ref_model.fc2.linear.weight.size(1), \
                'ERROR: `aligned_model` and `ref_model` have different number of `fc2` input features.'
            fc2_w = aligned_model.fc2.linear.weight.t().view(D1, -1)
            fc2_w_ref = ref_model.fc2.linear.weight.t().view(D1, -1)
            fc2_distances = torch.cdist(fc2_w_ref, fc2_w, p=2)
            sorted_idxs_fc2 = self.matching_method.match(fc2_distances)

            aligned_model = self.align_fc2_layer(aligned_model, sorted_idxs_fc2)

            # Sort the second (from the top) fully connected layer.
            D2, D3, K1, K2 = aligned_model.conv2[0].weight.size()
            assert D2 == ref_model.conv2[0].weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `fc1` input channels.'
            fc1_w = aligned_model.fc1.linear.weight.view(D1, D2, K1, K2).transpose(1, 0).reshape(D2, -1)
            fc1_w_ref = ref_model.fc1.linear.weight.view(D1, D2, K1, K2).transpose(1, 0).reshape(D2, -1)
            fc1_distances = torch.cdist(fc1_w_ref, fc1_w, p=2)
            sorted_idxs_fc1 = self.matching_method.match(fc1_distances)

            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)

            # Sort the first (from the top) convolutional layer.
            assert D3 == ref_model.conv2[0].weight.size(1), \
                'ERROR: `aligned_model` and `ref_model` have different number of `conv2` input channels.'
            conv2_w = aligned_model.conv2[0].weight.transpose(1, 0).reshape(D3, -1)
            conv2_w_ref = ref_model.conv2[0].weight.transpose(1, 0).reshape(D3, -1)
            conv2_distances = torch.cdist(conv2_w_ref, conv2_w, p=2)
            sorted_idxs_conv2 = self.matching_method.match(conv2_distances)

            aligned_model = self.align_conv2_layer(aligned_model, sorted_idxs_conv2)

        elif isinstance(model, GenericMLP):
            num_layers = len(aligned_model.layer_sizes) - 1
            
            for i in range(num_layers - 1):
                fc_idx = num_layers - i
                 
                fc_w = getattr(aligned_model, f'fc{fc_idx}').linear.weight.t()
                fc_w_ref = getattr(ref_model, f'fc{fc_idx}').linear.weight.t()
                fc_distances = torch.cdist(fc_w_ref, fc_w, p=2)
                sorted_idxs_fc = self.matching_method.match(fc_distances)

                aligned_model = self.align_fc_layer(aligned_model, fc_idx, sorted_idxs_fc)

        elif isinstance(model, VGG):
            # Align the three FC layers at the top.
            for fc_idx in [3, 2]:
                fc_w = getattr(aligned_model, f'fc{fc_idx}').linear.weight.t()
                fc_w_ref = getattr(ref_model, f'fc{fc_idx}').linear.weight.t()
                fc_distances = torch.cdist(fc_w_ref, fc_w, p=2)
                sorted_idxs_fc = self.matching_method.match(fc_distances)

                aligned_model = self.align_fc_layer(aligned_model, fc_idx, sorted_idxs_fc)

            # Align the layer at the junction with the conv part.
            # Sort the second (from the top) fully connected layer.
            last_conv = f'conv{aligned_model.num_conv_layers}'
            D1 = aligned_model.fc2.linear.weight.size(1)
            D2, D3, _, _ = getattr(aligned_model, last_conv)[0].weight.size()
            assert D2 == getattr(ref_model, last_conv)[0].weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `fc1` input channels.'
            fc1_w = aligned_model.fc1.linear.weight.view(D1, D2, -1).transpose(1, 0).reshape(D2, -1)
            fc1_w_ref = ref_model.fc1.linear.weight.view(D1, D2, -1).transpose(1, 0).reshape(D2, -1)
            fc1_distances = torch.cdist(fc1_w_ref, fc1_w, p=2)   
            sorted_idxs_fc1 = self.matching_method.match(fc1_distances)
            
            # Align the junction layer.
            C_out, _, _, _ = getattr(aligned_model, last_conv)[0].weight.size()
            _, D = aligned_model.fc1.linear.weight.size()
            assert D == C_out
            aligned_model.fc1.linear.weight = Parameter(aligned_model.fc1.linear.weight[:, sorted_idxs_fc1])
            # Propagating the permutation to the next layer.
            getattr(aligned_model, last_conv)[0].weight = \
                    Parameter(getattr(aligned_model, last_conv)[0].weight[sorted_idxs_fc1])
            getattr(aligned_model, last_conv)[0].bias = Parameter(getattr(aligned_model, last_conv)[0].bias[sorted_idxs_fc1])
        else:
            raise TypeError(f'ERROR: Invalid model type {type(model)}')

        return aligned_model


class ResNet18TopDownWeightMatchingBasedAlignment(AlignmentMethod):
    def __init__(self, matching_method):
        super().__init__()
        self.matching_method = matching_method


    def sort_bn_layer(self, bn, sorted_idxs):
        bn.running_mean = bn.running_mean[sorted_idxs]
        bn.running_var = bn.running_var[sorted_idxs]
        bn.weight = Parameter(bn.weight[sorted_idxs])
        bn.bias = Parameter(bn.bias[sorted_idxs])


    @torch.no_grad()
    def sort_layers(self, model, ref_model):
        assert isinstance(model, ResNet) and isinstance(ref_model, ResNet)
        aligned_model = copy.deepcopy(model)

        fc_w = getattr(aligned_model, f'fc1').weight.t()
        fc_w_ref = getattr(ref_model, f'fc1').weight.t()
        fc_distances = torch.cdist(fc_w_ref, fc_w, p=2)
        sorted_idxs = self.matching_method.match(fc_distances)

        aligned_model.fc1.weight = Parameter(
                aligned_model.fc1.weight[:, sorted_idxs])
        # Propagating the permutation to the conv and batch norm layers below.
        c5 = aligned_model.conv5_x
        self.sort_bn_layer(c5[1].residual_function[4], sorted_idxs)
        c5[1].residual_function[3].weight = Parameter(
                c5[1].residual_function[3].weight[sorted_idxs, :, :, :])
        
        c5[1].residual_function[0].weight = Parameter(
                c5[1].residual_function[0].weight[:, sorted_idxs, :, :])

        self.sort_bn_layer(c5[0].shortcut[1], sorted_idxs)
        c5[0].shortcut[0].weight = Parameter(
                c5[0].shortcut[0].weight[sorted_idxs, :, :, :])

        self.sort_bn_layer(c5[0].residual_function[4], sorted_idxs)
        c5[0].residual_function[3].weight = Parameter(
                c5[0].residual_function[3].weight[sorted_idxs, :, :, :])

        return aligned_model


class BottomUpActivationMatchingBasedAlignment(BottomUpAlignmentMethod):

    def __init__(self, matching_method):
        super().__init__()
        self.matching_method = matching_method


    @torch.no_grad()
    def sort_layers(self, model, ref_model, records):
        batch_size = len(records)
        assert batch_size > 0, 'ERROR: Please specify at least one record to compute activations on.'

        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            self.elapsed_times = []
            # Sorting the first convolutional layer.
            D1 = aligned_model.conv1[0].weight.size(0)
            assert D1 == ref_model.conv1[0].weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `conv1` output channels.'
            c1, c1_ref = aligned_model.conv1(records), ref_model.conv1(records)
            conv1_a, conv1_a_ref = c1.view(batch_size, D1, -1), c1_ref.view(batch_size, D1, -1)
            # Size B x D1 x F -> D1 x B x F, with F the size of each feature map.
            conv1_a = conv1_a.transpose(0, 1).contiguous().view(D1, -1)
            conv1_a_ref = conv1_a_ref.transpose(0, 1).contiguous().view(D1, -1)
            #conv1_distances = torch.cdist(conv1_a_ref, conv1_a, p=2).sum(dim=0)
            conv1_distances = torch.cdist(conv1_a_ref, conv1_a, p=2)
            #print(conv1_distances.size())
            start_time = time.time()
            sorted_idxs_conv1 = self.matching_method.match(conv1_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_conv_layer(aligned_model, sorted_idxs_conv1) 

            # Sorting the second convolutional layer.
            D2 = aligned_model.conv2[0].weight.size(0)
            assert D2 == ref_model.conv2[0].weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `conv2` output channels.'
            c2, c2_ref = aligned_model.conv2(aligned_model.conv1(records)), \
                    ref_model.conv2(ref_model.conv1(records))
            conv2_a, conv2_a_ref = c2.view(batch_size, D2, -1), c2_ref.view(batch_size, D2, -1)
            conv2_a = c2.transpose(0, 1).contiguous().view(D2, -1)
            conv2_a_ref = c2_ref.transpose(0, 1).contiguous().view(D2, -1)
            conv2_distances = torch.cdist(conv2_a_ref, conv2_a, p=2)
            #conv2_distances = torch.cdist(conv2_a_ref, conv2_a, p=2).sum(dim=0)
            start_time = time.time()
            sorted_idxs_conv2 = self.matching_method.match(conv2_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_junction_layer(aligned_model, sorted_idxs_conv2)

            # Sorting the first fully connected layer.
            D3 = aligned_model.fc1.linear.weight.size(0)
            assert D3 == ref_model.fc1.linear.weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
            fc1, fc1_ref = aligned_model.fc1(aligned_model.partial_forward(records)), \
                    ref_model.fc1(ref_model.partial_forward(records))
            fc1_a, fc1_a_ref = fc1.view(batch_size, D3, -1), \
                    fc1_ref.view(batch_size, D3, -1)
            fc1_a = fc1_a.transpose(0, 1).contiguous().view(D3, -1)
            fc1_a_ref = fc1_a_ref.transpose(1, 0).contiguous().view(D3, -1)
            fc1_distances = torch.cdist(fc1_a_ref, fc1_a, p=2)
            #fc1_distances = torch.cdist(fc1_a_ref, fc1_a, p=2).sum(dim=0)
            start_time = time.time()
            sorted_idxs_fc1 = self.matching_method.match(fc1_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)

        elif isinstance(model, VGG):
            conv_layers, fc_layers = aligned_model.forward_per_layer(records)
            conv_layers_ref, fc_layers_ref = ref_model.forward_per_layer(records)

            # First align the convolutional layers.
            for i in range(len(conv_layers)):
                c, c_ref = conv_layers[i], conv_layers_ref[i]
                D = getattr(aligned_model, f'conv{i+1}')[0].weight.size(0)
                assert D == getattr(ref_model, f'conv{i+1}')[0].weight.size(0), \
                        f'ERROR: `aligned_model` and `ref_model` have different number of `conv{i}` output channels.'
                c_a, c_a_ref = c.view(batch_size, D, -1), c_ref.view(batch_size, D, -1)
                c_a = c_a.transpose(0, 1).contiguous().view(D, -1)
                c_a_ref = c_a_ref.transpose(0, 1).contiguous().view(D, -1)
                c_distances = torch.cdist(c_a_ref, c_a, p=2)
                sorted_idxs_c = self.matching_method.match(c_distances)

                if i < len(conv_layers) - 1:
                    aligned_model = self.align_conv_layer(aligned_model, sorted_idxs_c, i+1)
                else:
                    aligned_model = self.align_junction_layer(aligned_model, sorted_idxs_c, len(conv_layers))

            # Sort the first fully connected layers.
            for i in range(len(fc_layers)-1):
                D = getattr(aligned_model, f'fc{i+1}').linear.weight.size(0)
                assert D == getattr(ref_model, f'fc{i+1}').linear.weight.size(0), \
                    f'ERROR: `aligned_model` and `ref_model` have different number of fc{i+1} output channels.'
                fc, fc_ref = fc_layers[i], fc_layers_ref[i]
                fc_a, fc_a_ref = fc.view(batch_size, D, -1), \
                        fc_ref.view(batch_size, D, -1)
                fc_a = fc_a.transpose(0, 1).contiguous().view(D, -1)
                fc_a_ref = fc_a_ref.transpose(0, 1).contiguous().view(D, -1)
                fc_distances = torch.cdist(fc_a_ref, fc_a, p=2)
                sorted_idxs_fc = self.matching_method.match(fc_distances)

                aligned_model = self.align_fc_layer(aligned_model, i+1, sorted_idxs_fc)

        else:
            raise TypeError(f'ERROR: Unsupported model type {type(model)}')
        
        return aligned_model


def compute_correlations(a1, a2, device='cpu'):
    """
    a1 and a2 are tensors of sizes BxNxF, where:
    -   B is the number of records (length of the sample we want to compute 
        the correlation over)
    -   N is the number of units (either feature maps for CNNs or neurons for 
        MLPs)
    -   F is the length of the unit (either F>1 for CNNs or F=1 for MLPs)
    Returns the pairwise correlations (NxNxF matrix) between pixels in each
    unit.
    """
    assert a1.size() == a2.size()
    B, N, F = a1.size()
    num_features = min(50, F)
    corrs = torch.zeros(size=(N, N, num_features))
    features = np.random.randint(F, size=num_features)
    a1 += torch.FloatTensor(np.random.uniform(1e-8, size=a1.size())).to(device)
    a2 += torch.FloatTensor(np.random.uniform(1e-8, size=a2.size())).to(device)
    a1 = (a1 - a1.mean(dim=0)) / a1.std(dim=0)
    a2 = (a2 - a2.mean(dim=0)) / a2.std(dim=0)
    for i in range(N):
        for j in range(N):
            for k, f in enumerate(features):
                # Compute the correlation manually (much faster).
                a12 = torch.dot(a1[:, i, f], a2[:, j ,f]) / B
                corrs[i, j, k] = a12
    return corrs.mean(dim=2)


class BottomUpCorrelationMatchingBasedAlignment(BottomUpAlignmentMethod):

    def __init__(self, matching_method, device='cpu'):
        super().__init__()
        self.matching_method = matching_method
        self.device = device


    @torch.no_grad()
    def sort_layers(self, model, ref_model, records):
        batch_size = len(records)
        assert batch_size > 0, 'ERROR: Please specify at least one record to compute activations on.'

        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            self.elapsed_times = []

            # Sorting the first convolutional layer.
            D1 = aligned_model.conv1[0].weight.size(0)
            assert D1 == ref_model.conv1[0].weight.size(0), \
                'ERROR: `aligned_model` and `ref_model` have different number of `conv1` output channels.'
            c1, c1_ref = aligned_model.conv1(records), ref_model.conv1(records)
            conv1_a, conv1_a_ref = c1.view(batch_size, D1, -1), \
                    c1_ref.view(batch_size, D1, -1)
            start_time = time.time()
            conv1_corrs = compute_correlations(conv1_a_ref, conv1_a, self.device)
            sorted_idxs_conv1 = self.matching_method.match(-conv1_corrs)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_conv_layer(aligned_model, 
                    sorted_idxs_conv1, 1)

            # Sorting the second convolutional layer.
            D2 = aligned_model.conv2[0].weight.size(0)
            assert D2 == ref_model.conv2[0].weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `conv2` output channels.'
            c2, c2_ref = aligned_model.conv2(aligned_model.conv1(records)), \
                    ref_model.conv2(ref_model.conv1(records))
            conv2_a, conv2_a_ref = c2.view(batch_size, D2, -1), \
                    c2_ref.view(batch_size, D2, -1)
            start_time = time.time()
            conv2_corrs = compute_correlations(conv2_a_ref, conv2_a, self.device)
            sorted_idxs_conv2 = self.matching_method.match(-conv2_corrs)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_junction_layer(aligned_model, 
                    sorted_idxs_conv2)

            # Sorting the first fully connected layer.
            D3 = aligned_model.fc1.linear.weight.size(0)
            assert D3 == ref_model.fc1.linear.weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
            fc1 = aligned_model.fc1(aligned_model.partial_forward(records))
            fc1_ref = ref_model.fc1(ref_model.partial_forward(records))
            fc1_a, fc1_a_ref = fc1.view(batch_size, D3, -1), \
                    fc1_ref.view(batch_size, D3, -1)
            start_time = time.time()
            fc1_corrs = compute_correlations(fc1_a_ref, fc1_a, self.device)
            sorted_idxs_fc1 = self.matching_method.match(-fc1_corrs)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)

        elif isinstance(model, GenericMLP):
            fc = aligned_model.partial_forward(records)
            fc_ref = ref_model.partial_forward(records)
            for i in range(len(aligned_model.layer_sizes) - 2):
                fc_idx = i + 1
                fc_name = f'fc{fc_idx}'
                # Sort the fully connected layer.
                aligned_model_fc = getattr(aligned_model, fc_name)
                ref_model_fc = getattr(ref_model, fc_name)
                D = aligned_model_fc.linear.weight.size(0)
                assert D == ref_model_fc.linear.weight.size(0), \
                    f'ERROR: `aligned_model` and `ref_model` have different number of {fc_name} output channels.'
                fc, fc_ref = aligned_model_fc(fc), ref_model_fc(fc_ref)
                fc_a, fc_a_ref = fc.view(batch_size, D, -1), \
                        fc_ref.view(batch_size, D, -1)
                fc_corrs = compute_correlations(fc_a_ref, fc_a, self.device)
                sorted_idxs = self.matching_method.match(-fc_corrs)
                aligned_model = self.align_fc_layer(aligned_model, fc_idx, 
                        sorted_idxs) 
        else:
            raise TypeError(f'ERROR: Invalid model type {type(aligned_model)}')


        return aligned_model


class TopDownActivationMatchingBasedAlignment(TopDownAlignmentMethod):

    def __init__(self, matching_method):
        super().__init__()
        self.matching_method = matching_method


    @torch.no_grad()
    def sort_layers(self, model, ref_model, records):
        batch_size = len(records)
        assert batch_size > 0, 'ERROR: Please specify at least one record to compute activations on.'

        aligned_model = copy.deepcopy(model)

        if isinstance(model, CNNLarge):
            self.elapsed_times = []

            # Sorting the first fully connected layer.
            D3 = aligned_model.fc1.linear.weight.size(0)
            assert D3 == ref_model.fc1.linear.weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `fc1` output channels.'
            fc1 = aligned_model.fc1(aligned_model.partial_forward(records))
            fc1_ref = ref_model.fc1(ref_model.partial_forward(records))
            fc1_a, fc1_a_ref = fc1.view(batch_size, D3, -1), \
                    fc1_ref.view(batch_size, D3, -1)
            fc1_a = fc1_a.transpose(0, 1).contiguous().view(D3, -1)
            fc1_a_ref = fc1_a_ref.transpose(1, 0).contiguous().view(D3, -1)
            fc1_distances = torch.cdist(fc1_a_ref, fc1_a, p=2)
            #fc1_distances = torch.cdist(fc1_a_ref, fc1_a, p=2).sum(dim=0)
            start_time = time.time()
            sorted_idxs_fc2 = self.matching_method.match(fc1_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_fc2_layer(aligned_model, sorted_idxs_fc2)

            # Sorting the second convolutional layer.
            D2 = aligned_model.conv2[0].weight.size(0)
            assert D2 == ref_model.conv2[0].weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `conv2` output channels.'
            c2, c2_ref = aligned_model.conv2(aligned_model.conv1(records)), \
                    ref_model.conv2(ref_model.conv1(records))
            conv2_a, conv2_a_ref = c2.view(batch_size, D2, -1), \
                    c2_ref.view(batch_size, D2, -1)
            conv2_a = c2.transpose(0, 1).contiguous().view(D2, -1)
            conv2_a_ref = c2_ref.transpose(0, 1).contiguous().view(D2, -1)
            conv2_distances = torch.cdist(conv2_a_ref, conv2_a, p=2)
            #conv2_distances = torch.cdist(conv2_a_ref, conv2_a, p=2).sum(dim=0)
            start_time = time.time()
            sorted_idxs_fc1 = self.matching_method.match(conv2_distances)
            self.elapsed_times.append(time.time()-start_time)

            aligned_model = self.align_fc1_layer(aligned_model, sorted_idxs_fc1)

            # Sorting the first convolutional layer.
            D1 = aligned_model.conv1[0].weight.size(0)
            assert D1 == ref_model.conv1[0].weight.size(0), \
                    'ERROR: `aligned_model` and `ref_model` have different number of `conv1` output channels.'
            c1, c1_ref = aligned_model.conv1(records), ref_model.conv1(records)
            conv1_a, conv1_a_ref = c1.view(batch_size, D1, -1), \
                    c1_ref.view(batch_size, D1, -1)
            # Size BxD1xF -> D1 x B x F, with F the size of each feature map.
            conv1_a = conv1_a.transpose(0, 1).contiguous().view(D1, -1)
            conv1_a_ref = conv1_a_ref.transpose(0, 1).contiguous().view(D1, -1)
            #conv1_distances = torch.cdist(conv1_a_ref, conv1_a, p=2).sum(dim=0)
            conv1_distances = torch.cdist(conv1_a_ref, conv1_a, p=2)
            #print(conv1_distances.size())
            start_time = time.time()
            sorted_idxs_conv2 = self.matching_method.match(conv1_distances)
            self.elapsed_times.append(time.time()-start_time) 

            aligned_model = self.align_conv2_layer(aligned_model, 
                    sorted_idxs_conv2)
        elif isinstance(model, GenericMLP):
            num_layers = len(aligned_model.layer_sizes) - 1

            # Align the layers from top to bottom.
            for i in range(num_layers - 1):
                fc_idx = num_layers - 1 - i
                # Compute the activations up to layer fc_idx.
                fc = aligned_model.partial_forward(records)
                fc_ref = ref_model.partial_forward(records)
                for j in range(fc_idx):
                    fc = getattr(aligned_model, f'fc{j+1}')(fc)
                    fc_ref = getattr(ref_model, f'fc{j+1}')(fc_ref)
                fc = fc.transpose(0, 1).contiguous()
                fc_ref = fc_ref.transpose(0, 1).contiguous()
                fc_distances = torch.cdist(fc_ref, fc, p=2)
                sorted_idxs_fc = self.matching_method.match(fc_distances)
                
                aligned_model = self.align_fc_layer(aligned_model, fc_idx+1, 
                        sorted_idxs_fc)
        else:
            raise TypeError(f'ERROR: Invalid model type {type(aligned_model)}')

        return aligned_model


class Matching(object):

    def match(self, distances):
        raise NotImplementedError


class GreedyMatching(Matching):

    def match(self, distances):
        #distances = distances.cpu().numpy()
        num_maps = len(distances)
        sorted, sorted_cols = torch.sort(distances, dim=1, stable=True)
        #print('sorted', sorted)
        #print('sorted_cols', sorted_cols)
        _, sorted_rows = torch.sort(sorted[:, 0], stable=True)
        #print('sorted_rows', sorted_rows)
        sorted_rows, sorted_cols = sorted_rows.cpu().numpy(), sorted_cols.cpu().numpy()
        already_matched = set()
        matching = np.zeros(num_maps)
        for i in sorted_rows:
            for j in range(num_maps):
                if sorted_cols[i][j] not in already_matched:
                    already_matched.add(sorted_cols[i][j])
                    matching[i] = sorted_cols[i][j]
                    break
        return matching


class HungarianAlgorithmMatching(Matching):

    def match(self, distances):
        distances = distances.cpu().numpy()
        # An array of row indices and one of corresponding column indices 
        # giving the optimal assignment.
        row_ind, col_ind = linear_sum_assignment(distances)
        return col_ind

