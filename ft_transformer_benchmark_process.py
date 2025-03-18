import json
import pickle
import os
from pathlib import Path

import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer


def get_col_cat_count(data):
    if data is None:
        return []
    col_cat_count = []
    for col in range(data.shape[1]):
        unique = np.unique(data[:, col])
        col_cat_count.append(len(unique))
    return col_cat_count


def main():
    seed = 0
    datasets = os.listdir('data/')
    for dataset in tqdm(datasets):
        files = os.listdir(Path('data', dataset))

        train_categorical = np.load(Path('data', dataset, 'C_train.npy')) if 'C_train.npy' in files else None
        train_numerical = np.load(Path('data', dataset, 'N_train.npy')) if 'N_train.npy' in files else None
        valid_categorical = np.load(Path('data', dataset, 'C_val.npy')) if 'C_val.npy' in files else None
        valid_numerical = np.load(Path('data', dataset, 'N_val.npy')) if 'N_val.npy' in files else None
        test_categorical = np.load(Path('data', dataset, 'C_test.npy')) if 'C_test.npy' in files else None
        test_numerical = np.load(Path('data', dataset, 'N_test.npy')) if 'N_test.npy' in files else None

        # Label encoding
        if train_categorical is not None:
            for i in range(train_categorical.shape[1]):
                le = LabelEncoder()
                train_categorical[:, i] = le.fit_transform(train_categorical[:, i])
                valid_categorical[:, i] = le.transform(valid_categorical[:, i])
                test_categorical[:, i] = le.transform(test_categorical[:, i])

        if train_numerical is not None:
            # Fill missing values with mean
            col_mean = np.nanmean(train_numerical, axis=0)
            inds = np.where(np.isnan(train_numerical))
            train_numerical[inds] = np.take(col_mean, inds[1])

            # Feature transformation
            temp = train_numerical.copy()
            if dataset in ["aloi", "helena"]:
                transformer = StandardScaler()
            else:
                noise = 1e-3
                stds = np.std(temp, axis=0, keepdims=True)
                noise_std = noise / np.maximum(stds, noise)
                temp += noise_std * np.random.default_rng(seed).standard_normal(temp.shape)
                transformer = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=max(min(temp.shape[0] // 30, 1000), 10),
                    subsample=int(1e9),
                    random_state=seed,
                )
            transformer.fit(temp)
            train_numerical = transformer.transform(train_numerical)
            valid_numerical = transformer.transform(valid_numerical)
            test_numerical = transformer.transform(test_numerical)

        if train_categorical is not None and train_numerical is not None:
            x_train = np.concatenate((train_categorical, train_numerical), axis=1).astype(np.float32)
            x_valid = np.concatenate((valid_categorical, valid_numerical), axis=1).astype(np.float32)
            x_test = np.concatenate((test_categorical, test_numerical), axis=1).astype(np.float32)
        elif train_categorical is not None:
            x_train = train_categorical.astype(np.float32)
            x_valid = valid_categorical.astype(np.float32)
            x_test = test_categorical.astype(np.float32)
        elif train_numerical is not None:
            x_train = train_numerical.astype(np.float32)
            x_valid = valid_numerical.astype(np.float32)
            x_test = test_numerical.astype(np.float32)
        else:
            raise ValueError('No data found')

        y_train = np.load(Path('data', dataset, 'y_train.npy'))
        y_valid = np.load(Path('data', dataset, 'y_val.npy'))
        y_test = np.load(Path('data', dataset, 'y_test.npy'))

        # Target transformation
        target_transformer = StandardScaler()
        y_train_transform = target_transformer.fit_transform(y_train.reshape(-1, 1)).squeeze()
        y_valid_transform = target_transformer.transform(y_valid.reshape(-1, 1)).squeeze()
        y_test_transform = target_transformer.transform(y_test.reshape(-1, 1)).squeeze()

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
