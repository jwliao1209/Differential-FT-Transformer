from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime

import numpy as np
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.preprocessing import QuantileTransformer

from src.dataset import TabularDataset, collate_fn
from src.models import get_model
from src.metric import cls_eval_funs, reg_eval_funs
from src.trainer import Trainer
from src.utils import load_pkl_data, set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--project_name', type=str, default='DOFEN_exp')
    parser.add_argument('--data_dir', type=str, default='/home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data')
    parser.add_argument('--data_id', type=str, default='361060')
    parser.add_argument('--config', type=str, default='doformer')
    parser.add_argument('--n_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument('--note', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def prepare_dataloader(data_dir: str, data_id: str, data_config, batch_size: int):
    data_dict = load_pkl_data(Path(data_dir, data_id, '0.pkl'))
    n_class = data_dict['label_cat_count']
    task = 'r' if n_class == -1 else 'c'
    target_transform = data_config.target_transform if task == 'r' else False
    train_X = data_dict['x_train']
    train_y = data_dict['y_train' if not target_transform else 'y_train_transform']
    valid_X = data_dict['x_val']
    valid_y = data_dict['y_val' if not target_transform else 'y_val_transform']
    test_X = data_dict['x_test']
    test_y = data_dict['y_test' if not target_transform else 'y_test_transform']

    data_args = {
        'task': task,
        'n_feature': train_X.shape[1],
        'n_train': train_X.shape[0],
        'n_valid': valid_X.shape[0],
        'n_test': test_X.shape[0],
    }
    col_cat_count = data_dict['col_cat_count']

    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)

    train_X_quantile = qt.fit_transform(data_dict['x_train_raw'])
    valid_X_quantile = qt.transform(data_dict['x_val_raw'])
    test_X_quantile = qt.transform(data_dict['x_test_raw'])

    train_set = TabularDataset(X=train_X, y=train_y, quantile=train_X_quantile)
    valid_set = TabularDataset(X=valid_X, y=valid_y, quantile=valid_X_quantile)
    test_set = TabularDataset(X=test_X, y=test_y, quantile=test_X_quantile)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    return train_loader, [valid_loader], [test_loader], n_class, data_args, col_cat_count


def prepare_cross_table_dataloader(data_dir: str, batch_size: int):
    all_train_data = []
    all_valid_dataset = []
    all_test_dataset = []
    num_column = 26

    data_ids = [
        '361055', # 10
        '361060', # 7
        '361061', # 10
        '361062', # 26
        '361065', # 10
        '361069'  # 24
    ]
    data_dir = '/home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data'

    PAD_VALUE = 0

    for data_id in data_ids:
        data_dict = load_pkl_data(Path(data_dir, data_id, '0.pkl'))

        train_X = data_dict['x_train']
        train_y = data_dict['y_train']

        train_mask = np.zeros((train_X.shape[0], num_column))
        train_mask[:train_X.shape[0], :train_X.shape[1]] = True

        train_X = np.pad(
            train_X,
            pad_width=[(0, 0), (0, num_column - train_X.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )
        
        qt = QuantileTransformer(output_distribution='uniform', n_quantiles=100)
        train_quantile_X = qt.fit_transform(data_dict['x_train_raw'])
        train_quantile_X = np.pad(
            train_quantile_X,
            pad_width=[(0, 0), (0, num_column - train_quantile_X.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )

        all_train_data.append((train_X, train_mask, train_quantile_X, train_y))

        valid_X = data_dict['x_val']
        valid_y = data_dict['y_val']

        valid_mask = np.zeros((valid_X.shape[0], num_column))
        valid_mask[:valid_X.shape[0], :valid_X.shape[1]] = True

        valid_X = np.pad(
            valid_X,
            pad_width=[(0, 0), (0, num_column - valid_X.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )

        valid_X_quantile = qt.transform(data_dict['x_val_raw'])
        valid_X_quantile = np.pad(
            valid_X_quantile,
            pad_width=[(0, 0), (0, num_column - valid_X_quantile.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )

        valid_set = TabularDataset(
            X=valid_X,
            y=valid_y,
            quantile=valid_X_quantile,
            mask=valid_mask,
        )

        all_valid_dataset.append(valid_set)

        test_X = data_dict['x_test']
        test_y = data_dict['y_test']

        test_mask = np.zeros((test_X.shape[0], num_column))
        test_mask[:test_X.shape[0], :test_X.shape[1]] = True

        test_X = np.pad(
            test_X,
            pad_width=[(0, 0), (0, num_column - test_X.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )

        test_X_quantile = qt.transform(data_dict['x_test_raw'])
        test_X_quantile = np.pad(
            test_X_quantile,
            pad_width=[(0, 0), (0, num_column - test_X_quantile.shape[1])],
            mode='constant',
            constant_values=PAD_VALUE,
        )

        test_set = TabularDataset(
            X=test_X,
            y=test_y,
            quantile=test_X_quantile,
            mask=test_mask,
        )

        all_test_dataset.append(test_set)

    all_train_X = np.concatenate([data[0] for data in all_train_data], axis=0)
    all_train_mask = np.concatenate([data[1] for data in all_train_data], axis=0)
    all_train_quantile = np.concatenate([data[2] for data in all_train_data], axis=0)
    all_train_y = np.concatenate([data[3] for data in all_train_data], axis=0)

    n_class = 2
    col_cat_count = [-1] * 26
    all_train_dataset = TabularDataset(
        X=all_train_X,
        y=all_train_y,
        quantile=all_train_quantile,
        mask=all_train_mask,
    )

    all_train_loader = DataLoader(
        all_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    all_valid_loader = [
        DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for valid_set in all_valid_dataset
    ]

    all_test_loader = [
        DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
        )
        for test_set in all_test_dataset
    ]

    data_args = {
        'task': 'c',
        'n_feature': all_train_X.shape[1],
        'n_train': all_train_X.shape[0],
    }

    return (
        all_train_loader,
        all_valid_loader,
        all_test_loader,
        n_class,
        data_args,
        col_cat_count,
    )


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(Path('configs', f'{args.config}.yaml'))
    set_random_seed()

    if args.data_id == 'cross_table':
        (
            train_loader,
            valid_loader,
            test_loader,
            n_class,
            data_args,
            col_cat_count,
        ) = prepare_cross_table_dataloader(args.data_dir, args.batch_size)
    else:
        (
            train_loader,
            valid_loader,
            test_loader,
            n_class,
            data_args,
            col_cat_count,
        ) = prepare_dataloader(args.data_dir, args.data_id, config.data, args.batch_size)

    model_params = {
        **config.model,
        'category_column_count': col_cat_count,
    }

    if n_class != -1:
        model_params['n_class'] = n_class

    model = get_model(
        model_name=config.model.name + data_args['task'],
        model_config=model_params,
    )


    # ###### Use pretrained model #######
    # checkpoint = torch.load('cross_table_model.pth')
    # model.load_state_dict(checkpoint)

    # for name, param in model.named_parameters():
    #     if 'rodt_forest_bagging' not in name:
    #         param.requires_grad = False

    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad = {param.requires_grad}")

    # import pdb; pdb.set_trace()
    # ####################################

    eval_funs = cls_eval_funs if data_args['task'] == 'c' else reg_eval_funs
    metrics = 'accuracy' if data_args['task'] == 'c' else 'r2'

    print("=" * 50)
    print("Model:", config.model.name)
    print("Number of features:", data_args['n_feature'])
    print("Number of training samples:", data_args['n_train'])
    print("Number of validation samples:", data_args['n_valid'])
    print("Number of test samples:", data_args['n_test'])
    print("Task:", data_args['task'])
    print("Number of classes:", n_class)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.n_epoch)
    print("Metrics:", metrics)
    print("=" * 50)

    if args.debug:
        wandb = None
    else:
        import wandb
        wandb.init(
            project=args.project_name,
            group=str(args.data_id),
            name=f"{args.data_id}_{config.model.name}_{datetime.today().strftime('%m%d_%H:%M:%S')}",
            config={'note': args.note} | vars(args) | data_args,
        )

    trainer = Trainer(
        model,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        eval_funs=eval_funs,
        metric=metrics,
        logger=wandb,
        record_best_performance=False if args.data_id == 'cross_table' else True,
        **config.trainer,
    )
    trainer.fit(
        train_loader=train_loader,
        valid_loaders=valid_loader,
        test_loaders=test_loader,
    )

if __name__ == '__main__':
    main()
