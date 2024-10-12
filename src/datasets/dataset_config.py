from os.path import join

_BASE_DATA_PATH = "../data"

dataset_config = {
    'iot_nidd': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'iot-nidd_dwn10p.parquet'),
    },
    'edge_iot': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                    'edge-iiot_dwn10p.parquet'),
    },
    'ton_iot': {  # class order
        'path': join(_BASE_DATA_PATH, 'uniform_label',
                     'ton-iot_dwn10p.parquet'),
    },
}

min_max_config = {
    'PL': (-1,1500), 'IAT': (-1,60000), 'DIR': (0,1), 'WIN': (-1,65535),
}
