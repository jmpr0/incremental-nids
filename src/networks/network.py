from copy import deepcopy

import numpy as np
import torch
from torch import nn
import math

from torch.utils.checkpoint import checkpoint
from functools import partial

class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model):
        head_var = model.head_var
        assert type(head_var) == str
        assert hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        self.feat_activation = nn.functional.relu

        last_layer = getattr(self.model, head_var)
        self.out_size = last_layer.in_features
        # converts last layer into identity
        # setattr(self.model, head_var, nn.Identity())
        # WARNING: this is for when pytorch version is <1.2
        setattr(self.model, head_var, nn.Sequential())

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs, **kwargs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])
        # print(self.task_offset)

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """        
        x = self.model.extract_features(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        y = []
        
        x_act = self.feat_activation(x)
        for head in self.heads:
            y.append(head(x_act))
        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')

    def unfreeze_all(self, t=0, freezing=None, verbose=True):
        """Unfreeze all parameters from the model, including the heads"""
        for name, param in self.named_parameters():
            if not freezing or not sum(nf in name for nf in freezing):
                param.requires_grad = True
        if verbose:
            self.trainability_info()

    def freeze_all(self, t=0, non_freezing=None, verbose=True):
        """Freeze all parameters from the model, including the heads"""
        for name, param in self.named_parameters():
            if not non_freezing or not sum(nf in name for nf in non_freezing):
                param.requires_grad = False
        if verbose:
            self.trainability_info()

    
    def trainability_info(self):
        print('\nTRAINABILITY INFO')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('')
    
    def freeze_conv(self):
        # self.model.conv1.requires_grad = False
        # self.model.bn1.requires_grad=False
        self.model.conv2.requires_grad=False
        self.model.bn2.requires_grad=False

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass

def get_padding(kernel, padding='same'):
    # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
    pad = kernel - 1
    if padding == 'same':
        if kernel % 2:
            return pad // 2, pad // 2
        else:
            return pad // 2, pad // 2 + 1
    return 0, 0

def compress_heads(state_dict):
    target_keys = sorted(
        [v for v in state_dict if 'heads' in v and '.0.' not in v])
    for target_key in target_keys:
        heads0 = 'heads.0.weight' if 'weight' in target_key else 'heads.0.bias'
        state_dict[heads0] = torch.cat(
            (state_dict[heads0], state_dict[target_key]))
        del state_dict[target_key]
    return state_dict

def get_output_dim(dimension, kernels, strides, dilatation=1, padding='same', return_paddings=False):
    # http://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
    out_dim = dimension
    paddings = []
    if padding == 'same':
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim + stride - 1) // stride
    else:
        for kernel, stride in zip(kernels, strides):
            paddings.append(get_padding(kernel, padding))
            out_dim = (out_dim - kernel + stride) // stride

    if return_paddings:
        return out_dim, paddings
    return out_dim
