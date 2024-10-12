import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loggers.exp_logger import ExperimentLogger


class Logger(ExperimentLogger):
    """Characterizes a disk logger"""

    def __init__(self, log_path, exp_name, begin_time=None):
        super(Logger, self).__init__(log_path, exp_name, begin_time)

        self.begin_time_str = '%d' % (begin_time * 1000)

        # Duplicate standard outputs
        sys.stdout = FileOutputDuplicator(sys.stdout,
                                          os.path.join(self.exp_path, 'stdout-{}.txt'.format(self.begin_time_str)), 'w')
        sys.stderr = FileOutputDuplicator(sys.stderr,
                                          os.path.join(self.exp_path, 'stderr-{}.txt'.format(self.begin_time_str)), 'w')

        # Raw log file
        self.raw_log_file = open(os.path.join(self.exp_path, "raw_log-{}.txt".format(self.begin_time_str)), 'a')

    def log_scalar(self, task, iter, name, value, group=None):
        # Raw dump
        entry = {"task": task, "iter": iter, "name": name, "value": value, "group": group,
                 "time": self.begin_time_str}
        self.raw_log_file.write(json.dumps(entry, sort_keys=True) + "\n")
        self.raw_log_file.flush()

    def log_args(self, args):
        with open(os.path.join(self.exp_path, 'args-{}.txt'.format(self.begin_time_str)), 'w') as f:
            json.dump(args.__dict__, f, separators=(',\n', ' : '), sort_keys=True)

    def log_result(self, array, name, step):
        if array.ndim <= 1:
            array = array[None]
        np.savetxt(os.path.join(self.exp_path, 'results', '{}-{}.txt'.format(name, self.begin_time_str)),
                   array, '%.6f', delimiter='\t')

    def log_parquet(self, df, name, task, append=False):
        df_path = os.path.join(self.exp_path, 'results', '{}_{}-{}.parquet'.format(name, task, self.begin_time_str))
        
        if os.path.exists(df_path) and append:
            df_old = pd.read_parquet(df_path)
            df = pd.concat([df_old, df], ignore_index=True)
        
        df.to_parquet(df_path)

    def log_figure(self, name, iter, figure):
        figure.savefig(os.path.join(self.exp_path, 'figures',
                                    '{}_{}-{}.png'.format(name, iter, self.begin_time_str)))
        figure.savefig(os.path.join(self.exp_path, 'figures',
                                    '{}_{}-{}.pdf'.format(name, iter, self.begin_time_str)))
        plt.close(figure)

    def save_model(self, state_dict, task, discr=''):
        torch.save(state_dict,
                   os.path.join(self.exp_path, "models", "task{}{}-{}.ckpt".format(task, discr, self.begin_time_str)))

    def __del__(self):
        self.raw_log_file.close()


class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
