import time
from argparse import ArgumentParser

import numpy as np
import torch
from datasets.exemplars_dataset import ExemplarsDataset
from loggers.exp_logger import ExperimentLogger
import pandas as pd


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None, **kwargs):
        self.model = model
        self._model = model
        
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        
        self.optimizer = None
        self.bias_layer = None
        self.is_train = True
        self.bias_layer = None
        self.__dict__.update(kwargs)
        
        self.initial_weights = self._model.get_copy()
 
    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None
    
    def refresh_initial_weights(self):
        self.initial_weights = self.model.get_copy()

    def _get_optimizer(self, t=None):
        """Returns the optimizer"""
        try:
            lr = self.lr if t is None or t > 0 else self.first_lr
        except:
            lr = self.lr
        try:
            momentum = self.momentum if t is None or t > 0 else self.first_momentum
        except:
            momentum = self.momentum
        return torch.optim.SGD(self._model.parameters(), lr=lr, weight_decay=self.wd, momentum=momentum)

    def train(self, t, trn_loader, val_loader):
        self.weights = dict()
        for k, l in self._model.state_dict().items():
            self.weights[k] = l.cpu().numpy()

        """Main train structure"""
        trainclock0 = time.time()
        self.pre_train_process(t, trn_loader)
        trainclock1 = time.time()
        
        trn_loader = self.train_loop(t, trn_loader, val_loader)

        trainclock2 = time.time()
        self.post_train_process(t, trn_loader, val_loader)
        trainclock3 = time.time()
        print('| Train Time: pre_train={:5.3f}s, train={:5.3f}s, post_train={:5.3f}s |'.format(
            trainclock1 - trainclock0, trainclock2 - trainclock1, trainclock3 - trainclock2))

    def pre_train_process(self, t, trn_loader, val_loader=None):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        self.is_train = True
        try:
            lr = self.lr if t > 0 else self.first_lr
        except:
            lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer(t)

        if self.bias_layer is not None:
            self.bias_layer.optimizer = self.optimizer
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()

            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            
            print('| Epoch {:3d}, time={:5.3f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

			# Valid
            clock3 = time.time()
            valid_loss, valid_acc, _, _, _, _ = self.eval(t, val_loader)
            clock4 = time.time()
            print(' Valid: time={:5.3f}s loss={:.3f}, TAw acc={:5.3f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")
        
            # # Adapt learning rate - patience scheme - early stopping regularization
            if valid_loss < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                # if the loss does not go down, decrease patience
                print()
                
                patience -= 1
                if patience <= 0:
                    # if it runs out of patience, reduce the learning rate
                    try:
                        lr /= (self.lr_factor if t > 0 else self.first_lr_factor)
                    except:
                        lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    try:
                        if lr < (self.lr_min if t > 0 else self.first_lr_min):
                            # if the lr decreases below minimum, stop the training session
                            # lr = self.lr
                            # lr = self.lr
                            print()
                            break
                    except:
                        if lr < self.lr_min:
                            # if the lr decreases below minimum, stop the training session
                            # lr = self.lr
                            print()
                            break
                    # reset patience and recover best model so far to continue training
                    
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    self.model.set_state_dict(best_model)
            
                    
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
            
        self.model.set_state_dict(best_model)
        self.is_train = False
        return trn_loader

    def post_train_process(self, t, trn_loader, val_loader=None):
        """Runs after training all the epochs of the task (after the train session)"""
        # TODO: if required
        if len(self.exemplars_dataset) > 0:
            # print("Incremental Learning - Post Processing")
            
            trn_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
            if self.bias_layer is not None:
                self.bias_layer.handle_head(self.model.task_cls)
                self.bias_layer.train_loop(t, trn_loader, self.device, self.model)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self._model.train()

        for images, targets in trn_loader:
            images, targets = self.format_inputs(images, targets)
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            loss = self.criterion(t, outputs, targets, features)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.clipgrad)
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
                outputs_tot.extend(np.concatenate(outputs, axis=1).tolist())
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

    def calculate_metrics(self, outputs, targets):
        targets = self.format_inputs(targets)
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self._model.task_cls.cumsum(0) <= tensor_to_cpu(targets[m])).sum()
            pred[m] = outputs[this_task][m].argmax() + self._model.task_offset[this_task]
        hits_taw = (pred == targets).float()
        # Task-Agnostic Multi-Head
        pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets=None, features=None):
        """Returns the loss value"""
        # print("CRITERION INC LEARNING")
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

    def format_inputs(self, images, targets=None):
        return format_inputs(self.device, images, targets)

    def save_logits(self, t, tst_loader, seed=42, is_train=True):
        print('Saving logits...')

        if self.exemplars_dataset is not None and len(self.exemplars_dataset):
            trn_loader = torch.utils.data.DataLoader(self.exemplars_dataset,
                                                    batch_size=tst_loader[t].batch_size,
                                                    shuffle=True,
                                                    num_workers=tst_loader[t].num_workers,
                                                    pin_memory=tst_loader[t].pin_memory)
            tst_loader.append(trn_loader)
        
        df_model = pd.DataFrame(columns=['Logits','Targets','is_train'])
       
        for i, dst in enumerate(tst_loader):
            images_tot = []
            model_out = []
            tar_list = []
            features_out = []
            for images, targets in dst:
                images, targets = format_inputs(self.device, images, targets)
                with torch.no_grad():
                    outputs, features = self.model(images, return_features=True)
                    images_tot.extend(np.array(tensor_to_cpu(images)).tolist())
                    model_out.extend(np.concatenate(tensor_to_cpu(outputs), axis=1).tolist())
                    tar_list.extend(targets.tolist())
                    features_out.extend(np.array(tensor_to_cpu(features)).tolist())
            df_model = df_model._append(pd.DataFrame({'Inputs': [images_tot],
                                                     'Logits': [model_out],
                                                     'Features': [features_out],
                                                     'Targets': [tar_list],
                                                     'is_train':is_train}), ignore_index=True)
        is_train_name = 'train' if is_train else 'test'
        self.logger.log_parquet(df_model, name=f'logits_features_targets_{is_train_name}_' + str(seed), task=t)


def format_inputs(device, images, targets=None):
    if isinstance(images, list):
        images = [v.to(device) for v in images]
    else:
        images = images.to(device)
    if targets is not None:
        targets = targets.to(device)
        return images, targets
    return images


def tensor_to_cpu(tensor):
    if isinstance(tensor, list):
        tensor = [v.cpu() for v in tensor]
    else:
        tensor = tensor.cpu()
    return tensor
