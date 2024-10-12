from copy import deepcopy
import torch
from torch.utils.data import DataLoader, Dataset
from .incremental_learning import Inc_Learning_Appr
import numpy as np


class Appr(Inc_Learning_Appr):
    """Class implementing the joint baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, logger=None, exemplars_dataset=None, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.trn_datasets = []
        self.val_datasets = []
        self.initial_weights = self.model.get_copy()

    def _has_exemplars(self):
        """Returns True in case exemplars are being used"""
        return self.exemplars_dataset is not None and len(self.exemplars_dataset) > 0

    def pre_train_process(self, t, trn_loader, val_loader=None):
        """Runs before training all epochs of the task (before the train session)"""

        # storing the weights for the current task head
        last_head_state_dict = self.model.heads[t].state_dict()
        self.initial_weights['heads.%d.weight' % t] = deepcopy(last_head_state_dict['weight'])
        self.initial_weights['heads.%d.bias' % t] = deepcopy(last_head_state_dict['bias'])
        # restore the initial weights
        self.model.set_state_dict(self.initial_weights)

        # continue pre-train as usual
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        
        self.trn_datasets.append(trn_loader.dataset)
        self.val_datasets.append(val_loader.dataset)

        trn_dset = JointDataset(self.trn_datasets)
        val_dset = JointDataset(self.val_datasets)
        
        trn_loader = DataLoader(trn_dset,
                                batch_size=trn_loader.batch_size,
                                shuffle=True,
                                num_workers=trn_loader.num_workers,
                                pin_memory=trn_loader.pin_memory)
        val_loader = DataLoader(val_dset,
                                batch_size=val_loader.batch_size,
                                shuffle=False,
                                num_workers=val_loader.num_workers,
                                pin_memory=val_loader.pin_memory)
        
        # continue training as usual
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader, correction=False):
        """Runs after training all the epochs of the task (after the train session)"""
        if t == 0 and len(self.trn_datasets) == 0:
            # Patch to correct the missing behavior when loading pre-trained base model
            self.trn_datasets.append(trn_loader.dataset)
            self.val_datasets.append(val_loader.dataset)

    def train_epoch(self, t, trn_loader, e=1, correction=False):
        """Runs a single epoch"""
        self.model.train()
        
        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs = self.model(images)
            
            loss = self.criterion(t, outputs, targets)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            
    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self._model.eval()
            for images, targets in val_loader:
                images, targets = self.format_inputs(images, targets)
                outputs, features = self.model(images, return_features=True)
                
                outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                features_tot.extend(features.tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

def tensor_to_cpu(tensor):
    if isinstance(tensor, list):
        tensor = [v.cpu() for v in tensor]
    else:
        tensor = tensor.cpu()
    return tensor

class JointDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset accumulates each task dataset incrementally"""

    def __init__(self, datasets):
        self.datasets = datasets
        self._len = sum([len(d) for d in self.datasets])

    def __len__(self):
        'Denotes the total number of samples'
        return self._len

    def __getitem__(self, index):
        for d in self.datasets:
            if len(d) <= index:
                index -= len(d)
            else:
                x, y = d[index]
                return x, y
