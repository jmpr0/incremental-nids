import argparse
import json
import os
import sys
from functools import partial
from glob import glob
from multiprocessing import Pool
import pyarrow

import numpy as np
import pandas as pd
from sklearn.metrics import *
from tqdm import tqdm
from scipy.stats.mstats import gmean

tqdm.pandas()

def compute_metrics(metric, target, score):

    target_indexes = sorted(set(target))
    score = np.array([[s[i] for i in target_indexes] for s in score])

    min_target = min(target_indexes)
    target = np.array([t - min_target for t in target])

    kwargs = dict()
    kwargs['y_true'] = target
    kwargs['y_pred'] = [np.argmax(s) for s in score]
    kwargs['average'] = None
    kwargs['zero_division'] = 0
    
    return globals()[metric](**kwargs)


def name_and_check_override(filename, discr, override):
    _filename = filename.replace('.parquet', '%s.parquet' % discr)
    if os.path.exists(_filename) and override is None:
        _override = input('File "%s" exists: override [Y, n]? ' % _filename).lower() != 'n'
    else:
        _override = override
    return _filename, _override


def main(df_filename, override=True):

    df = pd.read_parquet(df_filename[0])
    for df_fn in df_filename[1:]:
        df = pd.concat((df,pd.read_parquet(df_fn)), ignore_index=True)
    df.reset_index(inplace=True, drop=True)

    df_filename = '/'.join(df_filename[0].split('/')[:-1] + ['-'.join(
        ['_'.join(df_filename[0].split('/')[-1].split('-')[0].split('_')[:4]),  # Removing the eventual episode index
         df_filename[0].split('/')[-1].split('-')[-1]])])
    
    metrics_list = [
        'f1_score',
        'recall_score',
    ]

    print('Computing per-class metrics')
    df_per_class_metrics_filename, _override = name_and_check_override(df_filename, '_per_class_metrics', override)
    
    if not os.path.exists(df_per_class_metrics_filename) or _override:  # Generate or Override
        df_per_class_metrics = df.progress_apply(
            lambda x: dict([(metric, compute_metrics(metric, np.concatenate(x['Targets']), np.concatenate(x['Scores'])))
                            for metric in metrics_list]),
            axis=1, result_type='expand')
        df_per_class_metrics.to_parquet(df_per_class_metrics_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Incremental Learning Metrics Computer.')

    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)', nargs='*')
    parser.add_argument('--partial-exp-name', action='store_true',
                        help='If the exp-name should match as *exp-name* (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--yes', action='store_true',
                        help='Answer YES to all script requests (default=%(default)s)')
    parser.add_argument('--no', action='store_true',
                        help='Answer NO to all script requests (default=%(default)s)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (default=%(default)s)')

    args, _ = parser.parse_known_args(sys.argv)

    assert not (args.yes and args.no), 'YES and NO cannot be set together'

    override = args.yes or not args.no if (args.yes or args.no) else None

    exp_name_discr = '*' if args.partial_exp_name else ''

    df_filenames = []
    for exp_name in args.exp_name:
        fns = glob('%s/*%s%s/**/outputs_targets_*.parquet' % (args.results_path, exp_name, exp_name_discr),
                    recursive=True)
        df_filenames.extend([fn for fn in fns if 'metrics' not in fn and 'tsne' not in fn])

    # Group filenames by exp_dir and timestamp:
    # experiments falling in the same exp_dir MUST have a different timestamp
    df_filenames_dict = dict()
    for df_filename in df_filenames:
        key = (df_filename.split('/')[-3], df_filename.split('-')[-1])
        df_filenames_dict.setdefault(key, []).append(df_filename)

    # Sort filenames by episode
    for key in df_filenames_dict:
        episodes = [
            int(df_filename.split('/')[-1].split('-')[0].split('_')[-1]) for df_filename in df_filenames_dict[key]]
        sorting_index = np.argsort(episodes)
        df_filenames_dict[key] = [df_filenames_dict[key][i] for i in sorting_index]

    for df_filename in df_filenames_dict.values():
        main(df_filename, override)
