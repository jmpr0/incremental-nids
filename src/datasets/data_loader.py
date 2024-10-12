import numpy as np
from sklearn.utils import resample
from torch.utils import data

from . import networking_dataset as netdat
from .dataset_config import dataset_config


def get_loaders(datasets, num_tasks, nc_first_task, nc_incr_tasks, batch_size, num_workers, validation=.1,
                num_pkts=None, fields=None, seed=0):
    """Apply transformations to Datasets and create the DataLoaders for each task"""

    trn_load, val_load, tst_load = [], [], []
    taskcla = []
    labels_dicts=[]
    class_orders=[]
    dataset_offset = 0
    num_tasks = 1
    
    for idx_dataset, cur_dataset in enumerate(datasets, 0):
        # get configuration for current dataset
        dc = dataset_config[cur_dataset]
        
        trn_dset, val_dset, tst_dset, curtaskcla, labels_dict, class_order = get_datasets(
            cur_dataset, dc['path'], num_tasks, nc_first_task,
            nc_incr_tasks,
            validation=validation,
            class_order=None,
            num_pkts=num_pkts,
            fields=fields,
            seed=seed,
        )
        
        # apply offsets in case of multiple datasets
        labels_dicts.append(labels_dict)
        class_orders.append(class_order)
        
        if idx_dataset > 0:
            old_new_labels=dict([(i, i+dataset_offset) for i in np.arange(len(labels_dicts[-1]))])
            common_labels = set.intersection(set(labels_dicts[-2].keys()), set(labels_dicts[-1].keys()))              
            
            for l in common_labels:
                old_new_labels[labels_dicts[-1][l]]=labels_dicts[-2][l]
            for l in labels_dicts[-1].keys():
                if l not in common_labels:
                    value=labels_dicts[-1][l]
                    old_values=[labels_dicts[-1][k] for k in common_labels]
                    old_new_labels[labels_dicts[-1][l]]-=len([x for x in old_values if x<value])
            
            for tt in range(num_tasks):
                trn_dset[tt].labels=[old_new_labels[elem] for elem in trn_dset[tt].labels]
                val_dset[tt].labels=[old_new_labels[elem] for elem in val_dset[tt].labels]
                tst_dset[tt].labels=[old_new_labels[elem] for elem in tst_dset[tt].labels]
            
            print(np.unique(tst_dset[0].labels))
            class_orders[idx_dataset] = [old_new_labels[x] for x in class_orders[idx_dataset]]
            
            for tt in range(num_tasks):
                idx_trn=range(len(trn_dset[tt].labels))
                idx_val=range(len(val_dset[tt].labels))
                idx_tst=range(len(tst_dset[tt].labels))
                
                trn_dset[tt].labels=[trn_dset[tt].labels[i] for i in idx_trn]
                trn_dset[tt].images=[trn_dset[tt].images[i] for i in idx_trn]
                
                val_dset[tt].labels=[val_dset[tt].labels[i] for i in idx_val]
                val_dset[tt].images=[val_dset[tt].images[i] for i in idx_val]

                tst_dset[tt].labels=[tst_dset[tt].labels[i] for i in idx_tst]
                tst_dset[tt].images=[tst_dset[tt].images[i] for i in idx_tst]

            print(curtaskcla)
            curtaskcla=[(idx_dataset, len([x for x in class_orders[idx_dataset] if x>=dataset_offset]))]
            print(class_orders, curtaskcla)
                
        dataset_offset = dataset_offset + sum([tc[1] for tc in curtaskcla])

        # extend final taskcla list
        taskcla.extend(curtaskcla)

        # loaders
        for tt in range(num_tasks):
            trn_load.append(data.DataLoader(trn_dset[tt], batch_size=batch_size, shuffle=True, num_workers=num_workers))
            val_load.append(data.DataLoader(val_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers))
            tst_load.append(data.DataLoader(tst_dset[tt], batch_size=batch_size, shuffle=False, num_workers=num_workers))

    return trn_load, val_load, tst_load, taskcla


def get_datasets(dataset, path, num_tasks, nc_first_task, nc_incr_tasks, validation, class_order=None, num_pkts=None, fields=None, seed=0):
    """Extract datasets and create Dataset class"""

    trn_dset, val_dset, tst_dset = [], [], []
    labels_dict=None

    # read data paths and compute splits -- path needs to have a train.txt and a test.txt with image-label pairs
    all_data, taskcla, class_indices, labels_dict, class_order = netdat.get_data(
        path, num_tasks=num_tasks, nc_first_task=nc_first_task,
        nc_incr_tasks=nc_incr_tasks, validation=validation,
        shuffle_classes=class_order is None,
        class_order=class_order, num_pkts=num_pkts, fields=fields, seed=seed
    )

    # set dataset type
    Dataset = netdat.NetworkingDataset

    # get datasets, apply correct label offsets for each task
    offset = 0
    for task in range(num_tasks):
        all_data[task]['trn']['y'] = [label + offset for label in all_data[task]['trn']['y']]
        all_data[task]['val']['y'] = [label + offset for label in all_data[task]['val']['y']]
        all_data[task]['tst']['y'] = [label + offset for label in all_data[task]['tst']['y']]
        trn_dset.append(Dataset(all_data[task]['trn'], class_indices))
        val_dset.append(Dataset(all_data[task]['val'], class_indices))
        tst_dset.append(Dataset(all_data[task]['tst'], class_indices))
        offset += taskcla[task][1]

    if labels_dict is None:
        return trn_dset, val_dset, tst_dset, taskcla
    else:
        return trn_dset, val_dset, tst_dset, taskcla, labels_dict, class_order
