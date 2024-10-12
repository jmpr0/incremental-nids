import math
import os
import random

import numpy as np
import pandas as pd
import socket, struct
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import random
from .dataset_config import min_max_config


class NetworkingDataset(Dataset):
    """Characterizes a dataset for PyTorch -- this dataset pre-loads all biflows in memory"""

    def __init__(self, data, transform, class_indices=None):
        """Initialization"""
        self.labels = data['y']
        self.images = data['x']
        self.transform = transform
        self.class_indices = class_indices

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = self.images[index]
        y = self.labels[index]
        return x, y

    def set_modality(self, modality='all'):
        self.modality = modality

def normalize(x, n, field):
    x_n=np.zeros(x.shape)
    min_ = min_max_config[field][0]
    max_ = min_max_config[field][1]
    x_n = (x - min_) / (max_ - min_)
    x_n[x_n < 0] = 0
    x_n[x_n > 1] = 1
    return x_n

def get_data(full_path, num_tasks, nc_first_task, nc_incr_tasks, validation, shuffle_classes, 
             class_order=None, num_pkts=None, fields=None, seed=0):
    """Prepare data: dataset splits, task partition, class order"""

    data = {}
    taskcla = []

    if not os.path.exists(full_path):
        full_path = '../' + full_path

    path = os.path.join(*full_path.split('/')[:-1])
    dataset_filename = full_path.split('/')[-1]
    dataset_extension = dataset_filename.split('.')[-1]

    prep_df_path = os.path.join(path, dataset_filename.replace(
        '.%s' % dataset_extension, '_prep%d.%s' % (seed, dataset_extension)))
    all_fields = ['PL', 'IAT', 'DIR', 'WIN']
    
    df = None
    valid_idx = None
    
    if not os.path.exists(prep_df_path):
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import LabelEncoder

        if df is None:
            df = pd.read_parquet(full_path)
        
        all_fields = [f for f in all_fields if f in df]
        
        le = LabelEncoder()
        le.fit(df['LABEL'])
        df['ENC_LABEL'] = le.transform(df['LABEL'])
        np.savetxt(os.path.join(path, f"classes_{full_path.split('/')[-1].split('.')[0]}.txt"), le.classes_, fmt='%s')

        print('INFO: 0.7 train_ratio is applied.')
        train_idx, test_idx = [], []
        for lbl in df['LABEL'].unique():
            train_idx_tmp, test_idx_tmp = train_test_split(df[df['LABEL']==lbl].index, train_size=.7, random_state=seed)
            train_idx.extend(train_idx_tmp)
            test_idx.extend(test_idx_tmp)
        
        pad = 'FEAT_PAD' in df
        for field in all_fields:
            if pad:
                print('[INFO] Applying padding!')
                pad_value = 0.5 if field == 'DIR' else -1
                df[field] = df[[field, 'FEAT_PAD']].apply(
                    lambda x: np.concatenate((x[field], [pad_value] * x['FEAT_PAD'])), axis=1)

            df['SCALED_%s' % field] = df[field].apply(lambda x: normalize(x, num_pkts, field))
        df = df[['SCALED_%s' % field for field in all_fields] + ['ENC_LABEL']]
        
        df['IS_TRAIN'] = False
        df.loc[train_idx, 'IS_TRAIN'] = True
        
        df.to_parquet(prep_df_path)
    else:
        print('WARNING: using pre-processed dataframe.')
        df = pd.read_parquet(prep_df_path)
        train_idx, test_idx = df[df['IS_TRAIN']].index, df[~df['IS_TRAIN']].index

    print(df.columns)
    
    if class_order is None:
        num_classes = len(df['ENC_LABEL'].unique())
        class_order = list(range(num_classes))
    else:
        num_classes = len(class_order)
        class_order = class_order.copy()
    
    if shuffle_classes:
        np.random.shuffle(class_order)

    np.random.seed(seed)
    np.random.shuffle(class_order)

    print('Class order: ', class_order)
    labels=np.sort(np.unique(pd.read_parquet(full_path, columns=['LABEL'])['LABEL'].values))
    labels_dict=dict([(k, np.where(np.array(class_order)==i)[0][0]) for i, k in enumerate(labels)])
    
    # compute classes per task and num_tasks
    cpertask = np.array([num_classes])
    
    cpertask_cumsum = np.cumsum(cpertask)
    init_class = np.concatenate(([0], cpertask_cumsum[:-1]))

    # initialize data structure
    for tt in range(num_tasks):
        data[tt] = {}
        data[tt]['name'] = 'task-' + str(tt)
        data[tt]['trn'] = {'x': [], 'y': [], 's': []}
        data[tt]['val'] = {'x': [], 'y': [], 's': []}
        data[tt]['tst'] = {'x': [], 'y': [], 's': []}

    train_zip = zip(df.loc[train_idx, ['SCALED_%s' % field for field in fields]].values,
                    df.loc[train_idx, 'ENC_LABEL'], df.loc[train_idx].index)
    test_zip = zip(df.loc[test_idx, ['SCALED_%s' % field for field in fields]].values,
                    df.loc[test_idx, 'ENC_LABEL'], df.loc[test_idx].index)
    if valid_idx is not None:
        valid_zip = zip(df.loc[valid_idx, ['SCALED_%s' % field for field in fields]].values,
                    df.loc[valid_idx, 'ENC_LABEL'], df.loc[valid_idx].index)
        
    # ALL OR TRAIN
    for this_row in train_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]

        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, num_pkts, fields)

        data[this_task]['trn']['x'].append(this_sample)
        data[this_task]['trn']['y'].append(this_label - init_class[this_task])

    # ALL OR TEST
    for this_row in test_zip:
        this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
        this_label = int(this_label)
        
        if this_label not in class_order:
            continue
        # If shuffling is false, it won't change the class number
        this_label = class_order.index(this_label)

        # add it to the corresponding split
        this_task = (this_label >= cpertask_cumsum).sum()
        if this_task >= num_tasks:
            continue

        this_sample = format_data(this_biflow, num_pkts, fields)

        data[this_task]['tst']['x'].append(this_sample)
        data[this_task]['tst']['y'].append(this_label - init_class[this_task])

    # check classes
    for tt in range(num_tasks):
        data[tt]['ncla'] = len(np.unique(data[tt]['trn']['y']))
        assert data[tt]['ncla'] == cpertask[tt], "something went wrong splitting classes"

    # validation
    if valid_idx is None and validation > 0.0:
        for tt in data.keys():
            for cc in range(data[tt]['ncla']):
                cls_idx = list(np.where(np.asarray(data[tt]['trn']['y']) == cc)[0])
                rnd_img = random.sample(cls_idx, int(np.ceil(len(cls_idx) * validation)))
                rnd_img.sort(reverse=True)
                for ii in range(len(rnd_img)):
                    data[tt]['val']['x'].append(data[tt]['trn']['x'][rnd_img[ii]])
                    data[tt]['val']['y'].append(data[tt]['trn']['y'][rnd_img[ii]])
                    data[tt]['trn']['x'].pop(rnd_img[ii])
                    data[tt]['trn']['y'].pop(rnd_img[ii])
    else:
        # ALL OR VALID
        for this_row in valid_zip:
            this_biflow, this_label, this_ip = this_row[:-1], this_row[-2], this_row[-1]
            this_label = int(this_label)
            
            if this_label not in class_order:
                continue
            # If shuffling is false, it won't change the class number
            this_label = class_order.index(this_label)

            # add it to the corresponding split
            this_task = (this_label >= cpertask_cumsum).sum()
            if this_task >= num_tasks:
                continue

            this_sample = format_data(this_biflow, num_pkts, fields)

            data[this_task]['val']['x'].append(this_sample)
            data[this_task]['val']['y'].append(this_label - init_class[this_task])

    # other
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    
    return data, taskcla, class_order, labels_dict, class_order

def format_data(this_biflow, num_pkts, fields):
    this_sample = np.array([tb[:num_pkts] for tb in this_biflow[0]]).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')
    return this_sample

def get_random_state(idx):
    """
    Compute the random_state from the server IP address 
    """
    try:
        idx_family=socket.AF_INET6 if ':' in idx else socket.AF_INET
        fmt = '!L' if idx_family==socket.AF_INET else '!'+'L'*4
        server_ip = idx.split(',')[2]
    except:
        server_ip = idx
    
    try:
        return sum(struct.unpack(fmt, socket.inet_pton(idx_family, server_ip)))
    except:
        random_32bit_integer = random.randint(0, 2**32 - 1)
        return random_32bit_integer
