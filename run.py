from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime

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
    parser.add_argument('--model', type=str, default='diff')
    parser.add_argument('--norm', type=str, default='layer_norm')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_data(data_dir: str, data_id: str, data_config):
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

    return train_set, valid_set, test_set, n_class, data_args, col_cat_count


# def load_cross_table_data(data_dir: str, data_config):
#     data_ids = [
#         '361055', # 10
#         '361060', # 7
#         '361061', # 10
#         '361062', # 26
#         '361065', # 10
#         '361069', # 24
#     ]

#     n_class = 2
#     task = 'c'

#     train_X
#     train_y
#     valid_X
#     valid_y
#     test_X
#     test_y

#     data_args = {
#         'task': task,
#         'n_feature': 26,
#         'n_train': 0,
#         'n_valid': 0,
#         'n_test': 0,
#     }

#     col_cat_count = data_dict['col_cat_count']

#     for data_id in data_ids:
#         data_dict = load_pkl_data(Path(data_dir, data_id, '0.pkl'))
#         n_class = data_dict['label_cat_count']

#         train_X = data_dict['x_train']
#         train_y = data_dict['y_train']
#         valid_X = data_dict['x_val']
#         valid_y = data_dict['y_val']
#         test_X = data_dict['x_test']
#         test_y = data_dict['y_test']

#         data_args = {
#             'task': task,
#             'n_feature': train_X.shape[1],
#             'n_train': train_X.shape[0],
#             'n_valid': valid_X.shape[0],
#             'n_test': test_X.shape[0],
#         }
#         col_cat_count = data_dict['col_cat_count']


#     train_set = TabularDataset(X=train_X, y=train_y)
#     valid_set = TabularDataset(X=valid_X, y=valid_y)
#     test_set = TabularDataset(X=test_X, y=test_y)

#     return train_set, valid_set, test_set, n_class, data_args, col_cat_count


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(Path('configs', f'{args.model}.yaml'))
    set_random_seed()

    (
        train_set,
        valid_set,
        test_set,
        n_class,
        data_args,
        col_cat_count,
    ) = load_data(args.data_dir, args.data_id, config.data)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    model_params = {
        **config.model,
        'category_column_count': col_cat_count,
    }

    if n_class != -1:
        model_params['n_class'] = n_class

    model = get_model(args.model + data_args['task'])(**model_params)
    eval_funs = cls_eval_funs if data_args['task'] == 'c' else reg_eval_funs
    metrics = 'accuracy' if data_args['task'] == 'c' else 'r2'

    if args.norm == 'layer_norm':
        pass
    elif args.norm == 'dyt':
        from src.models.dyt import convert_ln_to_dyt
        model = convert_ln_to_dyt(model)
    elif args.norm == 'dyat':
        from src.models.dyt import convert_ln_to_dyat
        model = convert_ln_to_dyat(model)
    elif args.norm == 'fdyat':
        from src.models.dyt import convert_ln_to_fdyat
        model = convert_ln_to_fdyat(model)
    elif args.norm == 'dys':
        from src.models.dyt import convert_ln_to_dys
        model = convert_ln_to_dys(model)
    elif args.norm == 'dyas':
        from src.models.dyt import convert_ln_to_dyas
        model = convert_ln_to_dyas(model)
    else:
        raise ValueError(f"Invalid norm: {args.norm}")

    if args.debug:
        wandb = None
    else:
        import wandb
        wandb.init(
            project=args.project_name,
            group=str(args.data_id),
            name=f"{args.data_id}_{args.model}_{datetime.today().strftime('%m%d_%H:%M:%S')}",
            config=vars(args) | data_args,
        )

    trainer = Trainer(
        model,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        eval_funs=eval_funs,
        metric=metrics,
        logger=wandb,
        **config.trainer,
    )
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
    )

if __name__ == '__main__':
    main()
