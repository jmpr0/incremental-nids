import argparse
import importlib
import os
import time
from functools import reduce

import git
import numpy as np
import pandas as pd
import torch

import approach
import utils
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from loggers.exp_logger import MultiLogger
from networks import netmodels, nnmodels


def main(argv=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar='LOGGER')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar100'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar='DATASET')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=2, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=0, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--nc-incr-tasks', default=0, type=int, required=False,
                        help='Number of classes of the incremental tasks (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=2, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    parser.add_argument('--num-pkts', default=None, type=int,
                        help='Number of packets to select from the dataset')
    parser.add_argument('--fields', default=[], type=str, choices=['PL', 'IAT', 'DIR', 'WIN'],
                        help='Field or fields used (default=%(default)s)', nargs='+', metavar='FIELD')
    # model args
    parser.add_argument('--network', default=None, type=str, choices=nnmodels,
                        help='Network architecture used (default=%(default)s)', metavar='NETWORK')
    # training args
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar='APPROACH')
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--lr', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-7, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=10, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=20, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--validation', default=0.1, type=float, required=False,
                        help='Validation fraction (default=%(default)s)')

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)

    if args.nc_incr_tasks > 0:
        args.num_tasks = args.stop_at_task
    
    args.results_path = os.path.expanduser(args.results_path)

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)
    
    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    print('Extra Arguments =')
    print(extra_args)
    print('=' * 108)

    # Args -- CUDA
    device = 'cpu'

    ####################################################################################################################
    
    base_kwargs = dict(
        nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
        lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
        wd=args.weight_decay
    )

    # Args -- Store current commit hash
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_sha1 = repo.head.commit.hexsha
        setattr(args, 'git_sha1', repo_sha1)
    except:
        setattr(args, 'git_sha1', 'NO_REPO')

    # Args -- Network
    from networks.network import LLL_Net as Model

    net = getattr(importlib.import_module(name='networks'), args.network)
    init_model = net(num_pkts=args.num_pkts, num_fields=len(args.fields))
    
    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    assert len(extra_args) == 0, 'Unused args: {}'.format(' '.join(extra_args))

    ####################################################################################################################

    # Log all arguments
    try:
        full_exp_name = args.exp_name + '_' + args.approach + ('-mem' if appr_exemplars_dataset_args.num_exemplars else '')
    except:
        full_exp_name = args.exp_name + '_' + args.approach

    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(
        **args.__dict__,
        **appr_args.__dict__,
        **appr_exemplars_dataset_args.__dict__))

    # Loaders
    utils.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        args.datasets, args.num_tasks, args.nc_first_task, args.nc_incr_tasks, args.batch_size,
        validation=args.validation, num_workers=0, num_pkts=args.num_pkts, fields=args.fields, seed=args.seed,
    )
    
    # Apply arguments for loaders
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task
    print(f'{max_task=}')
    # Network and Approach instances
    utils.seed_everything(seed=args.seed)
    print(args.network)
    model = Model(init_model)
    print(model)
    utils.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    class_indices = first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(class_indices, **appr_exemplars_dataset_args.__dict__)

    utils.seed_everything(seed=args.seed)
    appr = Appr(model, device, **appr_kwargs)

    # Loop tasks
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        model.add_head(taskcla[t][1])
        model.to(device)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])

        appr.save_logits(t, trn_loader, args.seed)
        appr.save_logits(t, tst_loader, args.seed, is_train=False)

        print('-' * 108)
        
        # Test
        out_list = []
        tar_list = []
        features_list = []
        
        for u in range(t + 1):
            
            evalclock1 = time.time()
            test_loss, acc_taw[t, u], acc_tag[t, u], outputs, targets, features = appr.eval(u, tst_loader[u])
            evalclock2 = time.time()
            out_list.append(outputs)
            tar_list.append(targets)
            features_list.append(features)

            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.3f}%, forg={:5.3f}%'
                '| TAg acc={:5.3f}%, forg={:5.3f}%, time= {:5.3f}, Num_instances= {:2d} <<<'.format(
                u, test_loss, 100 * acc_taw[t, u], 100 * forg_taw[t, u], 100 * acc_tag[t, u], 100 * forg_tag[t, u],
                            evalclock2 - evalclock1, len(targets)))
            logger.log_scalar(task=t, iter=u, name='loss_' + str(args.seed), group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw_' + str(args.seed), group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag_' + str(args.seed), group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw_' + str(args.seed), group='test',
                            value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag_' + str(args.seed), group='test',
                            value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name='acc_taw_' + str(args.seed), step=t)
        logger.log_result(acc_tag, name='acc_tag_' + str(args.seed), step=t)
        logger.log_result(forg_taw, name='forg_taw_' + str(args.seed), step=t)
        logger.log_result(forg_tag, name='forg_tag_' + str(args.seed), step=t)
        try:
            logger.save_model(model.state_dict(), task=t)
        except:
            print('WARNING: model not saved.')
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1),
                        name='avg_accs_taw_' + str(args.seed), step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1),
                        name='avg_accs_tag_' + str(args.seed), step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name='wavg_accs_taw_' + str(args.seed), step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name='wavg_accs_tag_' + str(args.seed), step=t)
        
        # save scores, targets and features for each task
        df = pd.DataFrame({'Scores': [out_list], 'Targets': [tar_list], 'Features': [features_list]})
        logger.log_parquet(df, name='outputs_targets_features_' + str(args.seed), task=t)
        
    # Print Summary
    utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()
