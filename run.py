from argparse import ArgumentParser, Namespace
from pathlib import Path
from datetime import datetime

from src.models import get_model
from src.metric import cls_eval_funs, reg_eval_funs
from src.trainer import Trainer
from src.utils import load_pkl_data, set_random_seed


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/jiawei/Desktop/github/DOFEN/tabular-benchmark/tabular_benchmark_data')
    parser.add_argument('--data_id', type=str, default='361060')
    parser.add_argument('--model', type=str, default='dftt')
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=str, default=256)
    parser.add_argument('--target_transform', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    set_random_seed()

    data_dict = load_pkl_data(Path(args.data_dir, args.data_id, '0.pkl'))
    n_class = data_dict['label_cat_count']
    task = 'r' if n_class == -1 else 'c'

    target_transform = args.target_transform if task == 'r' else False
    train_X = data_dict['x_train']
    train_y = data_dict['y_train' if not target_transform else 'y_train_transform']
    test_X = data_dict['x_test']
    test_y = data_dict['y_test' if not target_transform else 'y_test_transform']

    model_params = {
        'category_column_count': data_dict['col_cat_count'],
    }
    if n_class != -1:
        model_params['n_class'] = n_class

    model = get_model(args.model + task)(**model_params)
    eval_funs = cls_eval_funs if task == 'c' else reg_eval_funs
    metrics = 'accuracy' if task == 'c' else 'r2'

    if args.debug:
        wandb = None
    else:
        import wandb
        wandb.init(
            project='DiffFTTransformer',
            group=str(args.data_id),
            name=f"{args.model}_{datetime.today().strftime('%m%d_%H:%M:%S')}",
            config=vars(args),
        )

    trainer = Trainer(
        model,
        batch_size=args.batch_size,
        n_epoch=args.n_epoch,
        eval_funs=eval_funs,
        metric=metrics,
        logger=wandb,
    )
    trainer.fit(
        train_X, train_y,
        test_X=test_X, test_y=test_y,
    )

if __name__ == '__main__':
    main()
