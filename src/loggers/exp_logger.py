import os
import importlib
from time import time


class ExperimentLogger:
    """Main class for experiment logging"""

    def __init__(self, log_path, exp_name, begin_time=None):
        self.log_path = log_path
        self.exp_name = exp_name
        self.exp_path = os.path.join(log_path, exp_name)
        if begin_time is None:
            self.begin_time = time() * 1000
        else:
            self.begin_time = begin_time

    def log_scalar(self, task, iter, name, value, group=None):
        pass

    def log_args(self, args):
        pass

    def log_result(self, array, name, step):
        pass

    def log_parquet(self, df, name, task, append=False):
        pass

    def log_figure(self, name, iter, figure):
        pass

    def save_model(self, state_dict, task):
        pass


class MultiLogger(ExperimentLogger):
    """This class allows to use multiple loggers"""

    def __init__(self, log_path, exp_name, loggers=None, save_models=True):
        super(MultiLogger, self).__init__(log_path, exp_name)
        if os.path.exists(self.exp_path):
            print("WARNING: {} already exists!".format(self.exp_path))
        os.makedirs(os.path.join(self.exp_path, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_path, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_path, 'figures'), exist_ok=True)

        self.save_models = save_models
        self.loggers = []
        for l in loggers:
            lclass = getattr(importlib.import_module(name='loggers.' + l + '_logger'), 'Logger')
            self.loggers.append(lclass(self.log_path, self.exp_name, self.begin_time))

    def log_scalar(self, task, iter, name, value, group=None):
        for l in self.loggers:
            l.log_scalar(task, iter, name, value, group)

    def log_args(self, args):
        for l in self.loggers:
            l.log_args(args)

    def log_result(self, array, name, step):
        for l in self.loggers:
            l.log_result(array, name, step)

    def log_parquet(self, df, name, task, append=False):
        for l in self.loggers:
            l.log_parquet(df, name, task, append)

    def log_figure(self, name, iter, figure):
        for l in self.loggers:
            l.log_figure(name, iter, figure)

    def save_model(self, state_dict, task, discr=''):
        if self.save_models:
            for l in self.loggers:
                l.save_model(state_dict, task, discr)
