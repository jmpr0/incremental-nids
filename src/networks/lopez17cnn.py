import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

from .network import get_output_dim

from torch.utils.checkpoint import checkpoint

class Lopez17CNN(nn.Module):
    """Lopez17CNN-like network for tests with HEADER input (n_inputs, n_features)."""

    def __init__(self, in_channels=1, num_classes=10, **kwargs):
        super().__init__()

        num_pkts = kwargs['num_pkts']
        num_fields = kwargs['num_fields']
        out_features_size = 200

        kernel = (4, 2)
        stride = (1, 1)
        self.pool_kernel0 = (3, 2)
        self.pool_kernel1 = (3, 1) if num_fields < 6 else self.pool_kernel0
        self.pool_stride = (1, 1)

        for padding in ['valid', 'same']:
            features_size0, self.paddings0 = get_output_dim(
                num_pkts,
                kernels=[kernel[0], self.pool_kernel0[0], kernel[0], self.pool_kernel1[0]],
                strides=[stride[0], self.pool_stride[0], stride[0], self.pool_stride[0]],
                padding=padding,
                return_paddings=True
            )
            features_size1, self.paddings1 = get_output_dim(
                num_fields,
                kernels=[kernel[1], self.pool_kernel0[1], kernel[1], self.pool_kernel1[1]],
                strides=[stride[1], self.pool_stride[1], stride[1], self.pool_stride[1]],
                padding=padding,
                return_paddings=True
            )
            if features_size0 < 1 or features_size1 < 1:
                pass
            else:
                print(f'{padding=}')
                break

        filters = [32, 64]

        # main part of the network
        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], kernel, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])
        self.fc1 = nn.Linear(features_size0 * features_size1 * filters[1], out_features_size)

        # last classifier layer (head) with as many outputs as classes
        self.fc = nn.Linear(out_features_size, num_classes)
        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc'

    def forward(self, x):
        out = F.relu(self.extract_features(x))
        out = self.fc(out)
        return out

    def extract_features_(self, x):
        out = checkpoint(self.extract_features_, x)
        return out

    def extract_features(self, x):
        out = F.pad(x, self.paddings1[0] + self.paddings0[0])
        out = F.relu(self.conv1(out))
        out = F.pad(out, self.paddings1[1] + self.paddings0[1])
        out = F.max_pool2d(out, self.pool_kernel0, stride=self.pool_stride, padding=0)
        out = self.bn1(out)
        out = F.pad(out, self.paddings1[2] + self.paddings0[2])
        out = F.relu(self.conv2(out))
        out = F.pad(out, self.paddings1[3] + self.paddings0[3])
        out = F.max_pool2d(out, self.pool_kernel1, stride=self.pool_stride, padding=0)
        out = self.bn2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        return out
