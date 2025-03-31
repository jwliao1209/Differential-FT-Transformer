from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf

from src.models import get_model
from src.metric import cls_eval_funs, reg_eval_funs
from src.trainer import Trainer
from src.utils import load_pkl_data, set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--project_name', type=str, default='DOFEN')
    parser.add_argument('--data_dir', type=str, default='/home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data')
    parser.add_argument('--data_id', type=str, default='361060')
    parser.add_argument('--model', type=str, default='diff')
    parser.add_argument('--norm', type=str, default='layer_norm')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument('--target_transform', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = OmegaConf.load(Path('configs', f'{args.model}.yaml'))
    set_random_seed()

    data_dict = load_pkl_data(Path(args.data_dir, args.data_id, '0.pkl'))
    n_class = data_dict['label_cat_count']
    task = 'r' if n_class == -1 else 'c'

    target_transform = args.target_transform if task == 'r' else False
    train_X = data_dict['x_train']
    train_y = data_dict['y_train' if not target_transform else 'y_train_transform']
    valid_X = data_dict['x_val']
    valid_y = data_dict['y_val' if not target_transform else 'y_val_transform']
    test_X = data_dict['x_test']
    test_y = data_dict['y_test' if not target_transform else 'y_test_transform']

    data_args = {
        'n_feature': train_X.shape[1],
        'n_train': train_X.shape[0],
        'n_valid': valid_X.shape[0],
        'n_test': test_X.shape[0],
    }

    model_params = {
        **config.model,
        'category_column_count': data_dict['col_cat_count'],
    }
    if n_class != -1:
        model_params['n_class'] = n_class

    model = get_model(args.model + task)(**model_params)
    eval_funs = cls_eval_funs if task == 'c' else reg_eval_funs
    metrics = 'accuracy' if task == 'c' else 'r2'

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
            name=f"{args.model}_{datetime.today().strftime('%m%d_%H:%M:%S')}",
            config=vars(args) | data_args,
        )

    trainer = Trainer(
        model,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        lr=config.trainer.lr,
        eval_funs=eval_funs,
        metric=metrics,
        logger=wandb,
    )
    trainer.fit(
        train_X=train_X, train_y=train_y,
        valid_X=valid_X, valid_y=valid_y,
        test_X=test_X, test_y=test_y,
    )

if __name__ == '__main__':
    main()
