import time
from argparse import ArgumentParser
from copy import deepcopy
import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from torch.utils.data import DataLoader
from .incremental_learning import Inc_Learning_Appr, tensor_to_cpu


class Appr(Inc_Learning_Appr):
    """Class implementing the Bias Correction (BiC) approach described in
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Large_Scale_Incremental_Learning_CVPR_2019_paper.pdf
    Original code available at https://github.com/wuyuebupt/LargeScaleIncrementalLearning
    """

    def __init__(self, model, device, nepochs=250, lr=0.1, lr_min=1e-5, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0.9, wd=0.0002, logger=None, exemplars_dataset=None, val_exemplar_percentage=0.1,
                 num_bias_epochs=200, T=2, **kwargs):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   logger, exemplars_dataset, **kwargs)
        self.val_percentage = val_exemplar_percentage
        self.bias_epochs = num_bias_epochs
        self.model_old = None
        self.T = T
        self.bias_layers = []

        self.x_valid_exemplars = []
        self.y_valid_exemplars = []

        self.x_validation_model = []
        self.y_validation_model = []
        self.loss_value = []
        self.loss_weights = []
        self.targets_new_task = []
        self.common_new = 0
        self.uncommon_new = 0
        
        if self.exemplars_dataset.max_num_exemplars != 0:
            self.num_exemplars = self.exemplars_dataset.max_num_exemplars
        elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
            self.num_exemplars_per_class = self.exemplars_dataset.max_num_exemplars_per_class

        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        assert (have_exemplars > 0), 'Error: BiC needs exemplars.'
        
    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--T', default=2, type=float, required=False,
                            help='Temperature scaling (default=%(default)s)')
        parser.add_argument('--val-exemplar-percentage', default=0.1, type=float, required=False,
                            help='Percentage of exemplars that will be used for validation (default=%(default)s)')
        parser.add_argument('--num-bias-epochs', default=200, type=int, required=False,
                            help='Number of epochs for training bias (default=%(default)s)')
        return parser.parse_known_args(args)

    def bias_forward(self, outputs):
        """Utility function --- inspired by https://github.com/sairin1202/BIC"""
        bic_outputs = []
        for m in range(len(outputs)):
            bic_outputs.append(self.bias_layers[m](outputs[m]))
        return bic_outputs

    def pre_train_process(self, t, trn_loader):
        # add a bias layer for the new classes
        print('PRE TRAIN PROCESS')
            
        self.bias_layers.append(BiasLayer().to(self.device))

        # STAGE 0: EXEMPLAR MANAGEMENT -- select subset of validation to use in Stage 2 -- val_old, val_new (Fig.2)
        print('Stage 0: Select exemplars from validation')
        clock0 = time.time()

        # number of classes and proto samples per class
        num_cls = sum(self.model.task_cls)
        num_old_cls = sum(self.model.task_cls[:t])
        if self.exemplars_dataset.max_num_exemplars != 0:
            num_exemplars_per_class = int(np.floor(self.num_exemplars / num_cls))
            num_val_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
            num_trn_ex_cls = num_exemplars_per_class - num_val_ex_cls
            # Reset max_num_exemplars
            self.exemplars_dataset.max_num_exemplars = (num_trn_ex_cls * num_cls).item()
        elif self.exemplars_dataset.max_num_exemplars_per_class != 0:
            num_val_ex_cls = int(np.ceil(self.val_percentage * self.num_exemplars_per_class))
            num_trn_ex_cls = self.num_exemplars_per_class - num_val_ex_cls
            # Reset max_num_exemplars
            self.exemplars_dataset.max_num_exemplars_per_class = num_trn_ex_cls

        # Remove extra exemplars from previous classes -- val_old
        if t > 0:
            if self.exemplars_dataset.max_num_exemplars != 0:
                num_exemplars_per_class = int(np.floor(self.num_exemplars / num_old_cls))
                num_old_ex_cls = int(np.ceil(self.val_percentage * num_exemplars_per_class))
                for cls in range(num_old_cls):
                    assert (len(self.y_valid_exemplars[cls]) == num_old_ex_cls)
                    self.x_valid_exemplars[cls] = self.x_valid_exemplars[cls][:num_val_ex_cls]
                    self.y_valid_exemplars[cls] = self.y_valid_exemplars[cls][:num_val_ex_cls]

        # Add new exemplars for current classes -- val_new
        non_selected = []
        for curr_cls in sorted(np.unique(trn_loader.dataset.labels)): #range(num_old_cls, num_cls):
            self.x_valid_exemplars.append([])
            self.y_valid_exemplars.append([])
            num_val_new_cls = num_val_ex_cls
            # get all indices from current class
            cls_ind = np.where(np.asarray(trn_loader.dataset.labels) == curr_cls)[0]
            # execute_revert_random_state(random.shuffle, dict(x=cls_ind), int(self.ls_factor))
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (num_val_new_cls <= len(cls_ind)), "Not enough samples to store for class {:d}".format(curr_cls)
            # add samples to the exemplar list
            self.x_valid_exemplars[curr_cls] = [trn_loader.dataset.images[idx] for idx in cls_ind[:num_val_new_cls]]
            self.y_valid_exemplars[curr_cls] = [trn_loader.dataset.labels[idx] for idx in cls_ind[:num_val_new_cls]]
            non_selected.extend(cls_ind[num_val_new_cls:])
        # remove selected samples from the validation data used during training
        trn_loader.dataset.images = [trn_loader.dataset.images[idx] for idx in non_selected]
        trn_loader.dataset.labels = [trn_loader.dataset.labels[idx] for idx in non_selected]
        clock1 = time.time()
        print(' > Selected {:d} validation exemplars, time={:5.3f}s'.format(
            sum([len(elem) for elem in self.y_valid_exemplars]), clock1 - clock0))

        # make copy to keep the type of dataset for Stage 2 -- not efficient
        self.bic_val_dataset = deepcopy(trn_loader.dataset)

        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop
        """
        # add exemplars to train_loader -- train_new + train_old (Fig.2)
        if t > 0:
                
            self.targets_new_task = np.unique(trn_loader.dataset.labels)
            self.common_new = len(np.intersect1d(self.targets_new_task, np.unique(self.exemplars_dataset.labels)))
            
            print(f'[INFO] ID: {id(trn_loader)} | New Classes: {self.targets_new_task} | Common Classes: {self.common_new}')
            # input()
            if not self.common_new:
                print('[WARNING] Training set and memory labels do not overlap')
                
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                    batch_size=trn_loader.batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    pin_memory=trn_loader.pin_memory)

        print('Stage 1: Training model with distillation')
        super().train_loop(t, trn_loader, val_loader)
        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader):
        print('POST TRAIN PROCESS')
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()
        
        # STAGE 2: BIAS CORRECTION
        if t > 0 and len(self.bias_layers)>0:
            print('Stage 2: Training bias correction layers')
            
            # Fill bic_val_loader with validation protoset
            if isinstance(self.bic_val_dataset.images, list):
                self.bic_val_dataset.images = sum(self.x_valid_exemplars, [])
            else:
                self.bic_val_dataset.images = np.vstack(self.x_valid_exemplars)

            self.bic_val_dataset.labels = sum(self.y_valid_exemplars, [])
            bic_val_loader = DataLoader(self.bic_val_dataset, batch_size=trn_loader.batch_size, shuffle=True,
                                        num_workers=0, pin_memory=trn_loader.pin_memory)
            n_tot_samples = len(bic_val_loader.dataset.labels)
            
            # bias optimization on validation
            self.model.eval()
            # Allow to learn the alpha and beta for the current task
            self.bias_layers[t].alpha.requires_grad = True
            self.bias_layers[t].beta.requires_grad = True

            # In their code is specified that momentum is always 0.9
            bic_optimizer = torch.optim.SGD(self.bias_layers[t].parameters(), lr=self.lr, momentum=0.9)
            # Loop epochs
            for e in range(self.bias_epochs):
                # Train bias correction layers
                clock0 = time.time()
                total_loss, total_acc = 0, 0
                for inputs, targets in bic_val_loader:
                    inputs, targets = self.format_inputs(inputs, targets)
                    # Forward current model
                    with torch.no_grad():
                        outputs = self.model(inputs)
                        old_cls_outs = self.bias_forward(outputs[:t])
                    new_cls_outs = self.bias_layers[t](outputs[t])
                    pred_all_classes = torch.cat([torch.cat(old_cls_outs, dim=1), new_cls_outs], dim=1)
                    # Eqs. 4-5: outputs from previous tasks are not modified (any alpha or beta from those is fixed),
                    #           only alpha and beta from the new task is learned. No temperature scaling used.
                    loss = torch.nn.functional.cross_entropy(pred_all_classes, targets)
                    # However, in their code, they apply a 0.1 * L2-loss to the gamma variable (beta in the paper)
                    loss += 0.1 * ((self.bias_layers[t].beta[0] ** 2) / 2)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc += ((pred_all_classes.argmax(1) == targets).float()).sum().item()
                    # Backward
                    bic_optimizer.zero_grad()
                    loss.backward()
                    bic_optimizer.step()
                clock1 = time.time()
                # reducing the amount of verbose
                if (e + 1) % (self.bias_epochs / 4) == 0:
                    print('| Epoch {:3d}, time={:5.3f}s | Train: loss={:.3f}, TAg acc={:5.3f}% |'.format(
                        e + 1, clock1 - clock0, total_loss / n_tot_samples,
                        100 * total_acc / n_tot_samples))
            # Fix alpha and beta after learning them
            self.bias_layers[t].alpha.requires_grad = False
            self.bias_layers[t].beta.requires_grad = False

        # Print all alpha and beta values
        if len(self.bias_layers)>0:
            for task in range(t + 1):
                print('Stage 2: BiC training for Task {:d}: alpha={:.5f}, beta={:.5f}'.format(
                    task, self.bias_layers[task].alpha.item(), self.bias_layers[task].beta.item()))

        # STAGE 3: EXEMPLAR MANAGEMENT
        if t >= 0:
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader, epoch=-1):
        """Runs a single epoch"""
        self.model.train()

        for i, (images, targets) in enumerate(trn_loader):
            images, targets = self.format_inputs(images, targets)
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images)
                targets_old = self.bias_forward(targets_old) if len(self.bias_layers) > 0 else targets_old # apply bias correction
                
            # Forward current model
            outputs = self.bias_forward(self.model(images))

            loss = self.criterion(t, outputs, targets, targets_old, epoch=epoch) 
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        outputs_tot, targets_tot, features_tot = [], [], []
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            
            for images, targets in val_loader:            
                images, targets = self.format_inputs(images, targets)
                # Forward old model
                targets_old = None

                if t > 0:
                    targets_old = self.model_old(images)
                    targets_old = self.bias_forward(targets_old) if len(self.bias_layers)>0 else targets_old  # apply bias correction

                # Forward current model
                outputs, features = self.model(images, return_features=True)
                features_tot.extend(features.tolist())
                outputs = self.bias_forward(outputs) if len(self.bias_layers)>0 else outputs # apply bias correction

                outputs_tot.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                targets_tot.extend(targets.tolist())
                loss = self.criterion(t, outputs, targets, targets_old, train=False)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, outputs_tot, targets_tot, features_tot

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        outputs /= self.T
        targets /= self.T

        tar = torch.nn.functional.softmax(targets, dim=1)
        return torch.nn.functional.cross_entropy(outputs, tar)

    def gkd(self, outputs_old, outputs, t):
        return self.cross_entropy(torch.cat(outputs[:t], dim=1), torch.cat(outputs_old[:t], dim=1), exp=1.0 / self.T)

    def criterion(self, t, outputs, targets, targets_old, epoch=-1, train=True):
        """Returns the loss value"""
        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if t > 0:
            loss_dist += float(self.gkd(targets_old, outputs, t))

        cat_outputs = torch.cat(outputs, dim=1)

        cce = torch.nn.functional.cross_entropy(cat_outputs, targets)

        lamb = (self.model.task_cls[:t].sum().float() / self.model.task_cls.sum()).to(self.device)
        return (1.0 - lamb) * cce + lamb * loss_dist
        
class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters"""

    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, requires_grad=False))
        self.beta = torch.nn.Parameter(torch.zeros(1, requires_grad=False))

    def forward(self, x):
        return self.alpha * x + self.beta
