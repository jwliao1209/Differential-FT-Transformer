from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        n_epoch: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        eval_funs: Optional[Dict[str, callable]] = None,
        metric: str = 'accuracy',
        logger: Optional[object] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_model: bool = False,
        early_stop_patience: int = float('inf'),
        verbose: bool = True,
    ) -> None:

        self.model = model
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.eval_funs = eval_funs
        self.metric = metric
        self.logger = logger
        self.save_model = save_model
        self.verbose = verbose
        self.early_stopping = EarlyStopping(patience=early_stop_patience)
        self.device = torch.device(device)
        self.set_optimization()

    def set_optimization(self) -> None:
        attn_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "attn" in name:
                attn_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {'params': attn_params, 'lr': 1e-4, 'weight_decay': self.weight_decay},
                {'params': other_params, 'lr': self.lr, 'weight_decay': self.weight_decay},
            ]
        )

    def train(self, train_loader: DataLoader) -> None:
        self.model.train()
        for data in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(
                X=data.X.to(self.device),
                y=data.y.to(self.device),
                quantile=data.quantile.to(self.device) if data.quantile is not None else None,
                mask=data.mask.to(self.device) if data.mask is not None else None,
            )
            outputs['loss'].backward()
            self.optimizer.step()

            if self.logger is not None:
                self.logger.log({'train_loss': outputs['loss'].item()})

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        for data in test_loader:
            pred = self.model.predict(
                X=data.X.to(self.device),
                quantile=data.quantile.to(self.device) if data.quantile is not None else None,
                mask=data.mask.to(self.device) if data.mask is not None else None,
            ).detach().cpu().numpy()
            preds.append(pred)
            labels.append(data.y)
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        evaluation_result = {}
        for metric, fun in self.eval_funs.items():
            evaluation_result[metric] = fun(preds, labels)
        return evaluation_result

    def fit(
        self,
        train_loader: DataLoader,
        valid_loaders: Optional[DataLoader] = None,
        test_loaders: Optional[DataLoader] = None,
    ) -> None:

        best_valid_epoch = 0
        best_valid_score = float('inf') if self.metric == 'rmse' else -float('inf')
        selected_test_score = float('inf') if self.metric == 'rmse' else -float('inf')

        self.model.to(self.device)
        pbar = trange(1, self.n_epoch + 1)
        for curr_epoch in pbar:
            self.train(train_loader)

            all_results = {'epoch': curr_epoch}

            if valid_loaders is not None:
                valid_results = {}
                for i, valid_loader in enumerate(valid_loaders):
                    results = self.test(valid_loader)
                    valid_results |= {f"{i}_valid_{metric}": value for metric, value in results.items()}

                # score = valid_results[f"valid_{self.metric}"]
                # if self.metric == 'rmse':
                #     self.early_stopping(score, 'min')
                #     if score < best_valid_score:
                #         best_valid_score = score
                #         best_valid_epoch = curr_epoch
                # else:
                #     self.early_stopping(score, 'max')
                #     if score > best_valid_score:
                #         best_valid_score = score
                #         best_valid_epoch = curr_epoch

                # valid_results |= {
                #     'best_valid_epoch': best_valid_epoch,
                #     'best_valid_score': best_valid_score,
                #     'early_stop_counter': self.early_stopping.counter,
                # }
                all_results |= valid_results

            if test_loaders is not None:
                test_results = {}
                for i, test_loader in enumerate(test_loaders):
                    results = self.test(test_loader)
                    test_results |= {f"{i}_test_{metric}": value for metric, value in results.items()}

                # if curr_epoch == best_valid_epoch:
                #     selected_test_score = test_results[f"test_{self.metric}"]

                # test_results |= {'selected_test_score': selected_test_score}
                all_results |= test_results

            pbar.set_postfix({'selected_test_score': selected_test_score})

            if self.verbose:
                tqdm.write(str(all_results))

            if self.logger is not None:
                self.logger.log(all_results)

            if self.early_stopping.early_stop:
                print("Stop training at epoch", curr_epoch)
                break

        self.model.cpu()

        if self.save_model:
            torch.save(self.model.state_dict(), 'model.pth')


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        verbose: bool = False,
    ) -> None:

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score: float, mode: str) -> None:
        score = -current_score if mode == 'min' else current_score

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1

            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                if self.verbose:
                    print('Early stopping triggered.')
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
