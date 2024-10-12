from itertools import cycle
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
from approach.incremental_learning import format_inputs
from datasets.exemplars_dataset import ExemplarsDataset
from networks.network import LLL_Net
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda
from sklearn.manifold import TSNE
import math

class ExemplarsSelector:
    """Exemplar selector for approaches with an interface of Dataset"""

    def __init__(self, exemplars_dataset: ExemplarsDataset):
        self.exemplars_dataset = exemplars_dataset
        self.already_added_classes = ()

    def __call__(self, model: LLL_Net, trn_loader: DataLoader, transform, t=None, from_inputs=False):
        tmp_model = deepcopy(model)
        # Management of the pre-allocated-output: the non used heads info are removed
        if t is not None:
            tmp_model._modules['heads'] = tmp_model._modules['heads'][:t + 1]
            tmp_model.task_cls = tmp_model.task_cls[:t + 1]
            tmp_model.task_offset = tmp_model.task_offset[:t + 1]
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(tmp_model)
        
        # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
        sel_loader = DataLoader(trn_loader.dataset, batch_size=trn_loader.batch_size, shuffle=False,
                                num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
        selected_indices = self._select_indices(
            tmp_model, sel_loader, exemplars_per_class, transform, from_inputs)
        self.already_added_classes = set(_get_labels(sel_loader))
        
        x, y = zip(*(trn_loader.dataset[idx] for idx in selected_indices))
        
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y

    def _exemplars_per_class_num(self, model: LLL_Net):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform, from_inputs=None) -> Iterable:
        pass


class RandomExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on random selection, which produces a random list of samples."""

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform, from_inputs=None) -> Iterable:
        
        labels = _get_labels(sel_loader)

        num_cls = sum(model.task_cls)
        result = []
        for curr_cls in range(self.exemplars_dataset.base_class_index, num_cls):
            # get all indices from current class -- check if there are exemplars from previous task in the loader
            cls_ind = np.where((labels == curr_cls))[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            if exemplars_per_class < len(cls_ind):
                # select the exemplars randomly
                result.extend(random.sample(list(cls_ind), exemplars_per_class))
            else:
                print('WARNING: Not enough samples to store for class {:d}: selected ALL.'.format(curr_cls))
                result.extend(list(cls_ind))
        return result


class HerdingExemplarsSelector(ExemplarsSelector):
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)
 
    def _select_indices(self, model: LLL_Net, sel_loader: DataLoader, exemplars_per_class: int, transform, from_inputs=False) -> Iterable:
        model_device = next(model.parameters()).device  # we assume here that whole model is on a single device

        # extract outputs from the model for all train samples
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                images = format_inputs(model_device, images)
                # The argument from_inputs is used to select inputs instead of features, if required
                if not from_inputs:
                    logits, feats = model(images, return_features=True)
                    feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                else:
                    feats = images
                extracted_features.append(feats)
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []
        # iterate through all classes
        for curr_cls in np.unique(extracted_targets):
            # get all indices from current class
            cls_ind = np.where((extracted_targets == curr_cls))[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            if exemplars_per_class < len(cls_ind):
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # calculate the mean
                cls_mu = cls_feats.mean(0)
                # select the exemplars closer to the mean of each class
                selected = []
                selected_feat = []
                for k in range(exemplars_per_class):
                    sum_others = torch.zeros(cls_feats.shape[1:])
                    for j in selected_feat:
                        sum_others += j / (k + 1)
                    newone = cls_ind[0]
                    newonefeat = extracted_features[newone]
                    dist_min = torch.norm(cls_mu - newonefeat / (k + 1) - sum_others)
                    # choose the closest to the mean of the current class
                    for item in cls_ind[1:]:
                        if item not in selected:
                            feat = extracted_features[item]
                            dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                            if dist < dist_min:
                                dist_min = dist
                                newone = item
                                newonefeat = feat
                    selected_feat.append(newonefeat)
                    selected.append(newone)
                result.extend(selected)
            else:
                print('WARNING: Not enough samples to store for class {:d}: selected ALL.'.format(curr_cls))
                result.extend(list(cls_ind))
        return result

def _get_labels(sel_loader):
    if hasattr(sel_loader.dataset, 'labels'):  # BaseDataset, MemoryDataset
        labels = np.asarray(sel_loader.dataset.labels)
    elif hasattr(sel_loader.dataset, 'datasets'):
        labels = []
        for ds in sel_loader.dataset.datasets:
            labels.extend(ds.labels)
        labels = np.array(labels)
    else:
        raise RuntimeError("Unsupported dataset: {}".format(sel_loader.dataset.__class__.__name__))
    return labels
