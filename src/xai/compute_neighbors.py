import pandas as pd
import numpy as np
from glob import glob
import json
from box import Box
from sklearn.metrics import *
from tqdm import tqdm
import os
import argparse

def get_appr_params(base_path, T=2):
    dict_appr_params  = {}
    alpha, beta = None, None
    results_files = [elem for elem in glob(f'{base_path}/results/*') if (('outputs_targets_features_' in elem) \
                        and elem.split('/')[-1].split('-')[0].split('_')[-1]=='1')]
    if len(results_files)==0:
        results_files = [elem for elem in glob(f'{base_path}/results/*') if 'per_class_metrics' in elem]    
    for out_path in results_files:
        timestamp = out_path.split('.')[-2].split('-')[-1].split('_')[0]
        args_file = f'{base_path}/args-{timestamp}.txt'
        stdout_file = f'{base_path}/stdout-{timestamp}.txt'
        with open(stdout_file) as f:
            lines = f.readlines()
        with open(args_file) as f:
            args = Box(json.loads(f.read()))
        t = float(args.get('T', default = 2))
        if t==T:
            class_order = ''
            for l in lines:
                if 'BiC training for Task 1' in l:
                    alpha=float(l.split('alpha=')[-1].split(',')[0])
                    beta=float(l.split('beta=')[-1])
                if '[[' in l and ']]' in l:
                    class_order=[int(elem) for elem in l.split(']]')[0].replace('[','').replace(']','').replace('','').split(',')]
                    class_order=[class_order[:args.nc_first_task], class_order[args.nc_first_task:]]
            dict_appr_params[args.seed]={'alpha':alpha, 'beta':beta,
                'timestamp':timestamp, 'class_order':class_order}
    return dict_appr_params


def compute_neighbors(sample_feats, dataset_src_feats, dataset_tgt_feats, k=5):
        top_k_nb = []
        for sample in tqdm(sample_feats):
            rows = []

            src_distances = np.linalg.norm(sample - dataset_src_feats, axis=1)
            tgt_distances = np.linalg.norm(sample - dataset_tgt_feats, axis=1)

            sortidx = np.argsort(np.concatenate((src_distances, tgt_distances)))[:k]
            values =  np.concatenate((np.full(len(src_distances), 'SRC'), np.full(len(tgt_distances), 'TGT')))[sortidx]

            top_k_nb.append(values)
        return top_k_nb


def main(appr_path, k):
    dict_bic = get_appr_params(appr_path)

    seed = 1
    ts = dict_bic[seed]['timestamp']

    args_path = f'{appr_path}/args-{ts}.txt'
    with open(args_path) as f:
        args = Box(json.loads(f.read()))

    assert args.seed == seed

    res = pd.read_parquet(f'{appr_path}/results/outputs_targets_features_{args.seed}_1-{ts}.parquet')

    true_src = res['Targets'][0][0].copy()
    true_tgt = res['Targets'][0][1].copy()

    score_src = res['Scores'][0][0].copy()
    score_tgt = res['Scores'][0][1].copy()

    features_src = res['Features'][0][0].copy()
    features_tgt = res['Features'][0][1].copy()

    score_src = np.array([[v for v in s] for s in score_src])
    score_tgt = np.array([[v for v in s] for s in score_tgt])

    features_src = np.array([[v for v in s] for s in features_src])
    features_tgt = np.array([[v for v in s] for s in features_tgt])

    pred_src = np.argmax(score_src, axis=1)
    pred_tgt = np.argmax(score_tgt, axis=1)

    df_trn = pd.read_parquet(f'{appr_path}/results/logits_features_targets_train_{args.seed}_1-{ts}.parquet')

    true_trn = df_trn['Targets'][0].copy()
    score_trn = df_trn['Logits'][0].copy()
    features_trn = df_trn['Features'][0].copy()
    true_mem = df_trn['Targets'][1].copy()
    score_mem = df_trn['Logits'][1].copy()
    features_mem = df_trn['Features'][1].copy()

    score_trn = np.array([[v for v in s] for s in score_trn])
    features_trn = np.array([[v for v in s] for s in features_trn])
    score_mem = np.array([[v for v in s] for s in score_mem])
    features_mem = np.array([[v for v in s] for s in features_mem])

    corr_feat_src = np.array([[f for f in feat] for feat, true, pred in zip(features_src, true_src, pred_src) if true==pred])
    corr_feat_tgt = np.array([[f for f in feat] for feat, true, pred in zip(features_tgt, true_tgt, pred_tgt) if true==pred])

    # Given a sample, top K neighbors between the incremental train set and memory
    # Found K neighbors, distribution of the top K between the datasets (src and tgt)

    top_k_nb_src = compute_neighbors(corr_feat_src, features_mem, features_trn, k=k)
    top_k_nb_tgt = compute_neighbors(corr_feat_tgt, features_mem, features_trn, k=k)

    outdir = f'./neighbors/' + appr_path.rstrip('/').split('/')[-1]
    os.makedirs(outdir, exist_ok=True)

    np.array([[v0 for v0 in v] for v in top_k_nb_src]).dump(f'{outdir}/top_{k}_nb_src.np')
    np.array([[v0 for v0 in v] for v in top_k_nb_tgt]).dump(f'{outdir}/top_{k}_nb_tgt.np')


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description='This script computes top K neighbors between the incremental training set and the memory.'
    )
    
    # Add arguments
    parser.add_argument(
        '-p', '--exp-path',
        type=str,
        required=True,
        help='Path to the experiment(s).'
    )

    # Add arguments
    parser.add_argument(
        '-k', '--num-neighbors',
        type=int,
        required=False,
        default=5,
        help='Number of neighbors to consider.'
    )

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with the parsed arguments
    main(args.exp_path, args.num_neighbors)
