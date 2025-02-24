import json
import pickle
import os
from pathlib import Path

import numpy as np
import sklearn
from tqdm import tqdm


def get_col_cat_count(data):
    if data is None:
        return []
    col_cat_count = []
    for col in range(data.shape[1]):
        unique = np.unique(data[:, col])
        col_cat_count.append(len(unique))
    return col_cat_count


def main():
    datasets = os.listdir('data/')

    for dataset in tqdm(datasets):
        files = os.listdir(Path('data', dataset))

        train_categorical = np.load(Path('data', dataset, 'C_train.npy')) if 'C_train.npy' in files else None
        train_numerical = np.load(Path('data', dataset, 'N_train.npy')) if 'N_train.npy' in files else None
        valid_categorical = np.load(Path('data', dataset, 'C_val.npy')) if 'C_val.npy' in files else None
        valid_numerical = np.load(Path('data', dataset, 'N_val.npy')) if 'N_val.npy' in files else None
        test_categorical = np.load(Path('data', dataset, 'C_test.npy')) if 'C_test.npy' in files else None
        test_numerical = np.load(Path('data', dataset, 'N_test.npy')) if 'N_test.npy' in files else None

        if train_categorical is not None and train_numerical is not None:
            x_train = np.concatenate((train_categorical, train_numerical), axis=1)
            x_valid = np.concatenate((valid_categorical, valid_numerical), axis=1)
            x_test = np.concatenate((test_categorical, test_numerical), axis=1)
        elif train_categorical is not None:
            x_train = train_categorical
            x_valid = valid_categorical
            x_test = test_categorical
        elif train_numerical is not None:
            x_train = train_numerical
            x_valid = valid_numerical
            x_test = test_numerical
        else:
            raise ValueError('No data found')

        y_train = np.load(Path('data', dataset, 'y_train.npy')).reshape(-1, 1)
        y_valid = np.load(Path('data', dataset, 'y_val.npy')).reshape(-1, 1)
        y_test = np.load(Path('data', dataset, 'y_test.npy')).reshape(-1, 1)

        transformer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(y_train.shape[0] // 30, 1000), 10),
            random_state=0,
        )
        y_train_transform = transformer.fit_transform(y_train.reshape(-1, 1))
        y_valid_transform = transformer.transform(y_valid.reshape(-1, 1))
        y_test_transform = transformer.transform(y_test.reshape(-1, 1))

        info_path = Path('data', dataset, 'info.json')
        with open(info_path, 'r') as f:
            info = json.load(f)

        data_dict = {}
        data_dict['x_train'] = x_train
        data_dict['y_train'] = y_train
        data_dict['y_train_transform'] = y_train_transform
        data_dict['x_val'] = x_valid
        data_dict['y_val'] = y_valid
        data_dict['y_val_transform'] = y_valid_transform
        data_dict['x_test'] = x_test
        data_dict['y_test'] = y_test
        data_dict['y_test_transform'] = y_test_transform

        if info['task_type'] == 'regression':
            data_dict['label_cat_count'] = -1
        elif info['task_type'] == 'binclass':
            data_dict['label_cat_count'] = 2
        elif info['task_type'] == 'multiclass':
            data_dict['label_cat_count'] = len(np.unique(y_train))

        data_dict['col_cat_count'] = get_col_cat_count(train_categorical) + [-1] * train_numerical.shape[1]

        data_dict['x_train_raw'] = None
        data_dict['x_val_raw'] = None
        data_dict['x_test_raw'] = None
        data_dict['target_transformer'] = None
        data_dict['dataset_config'] = None

        os.makedirs(Path('ft_transformer_benchmark', dataset), exist_ok=True)
        with open(Path('ft_transformer_benchmark', dataset, '0.pkl'), 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
