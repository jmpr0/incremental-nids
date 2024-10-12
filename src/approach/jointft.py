import torch
from datasets.exemplars_dataset import ExemplarsDataset
from .incremental_learning import Inc_Learning_Appr
import numpy as np


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, logger=None, exemplars_dataset=None, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.__dict__.update(kwargs)
    
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                batch_size=trn_loader.batch_size,
                                                shuffle=True,
                                                num_workers=trn_loader.num_workers,
                                                pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        return super().train_loop(t, trn_loader, val_loader)

    def post_train_process(self, t, trn_loader, val_loader, correction=False):
        """Runs after training all the epochs of the task (after the train session)"""
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
        return super().post_train_process(t, trn_loader, val_loader)
    
    def criterion(self, t, outputs, targets, features=None, epoch=1):
        """Returns the loss value"""
        # print("CRITERION JOINTFT")
        return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
