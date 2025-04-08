from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm

from .early_stop import EarlyStopping


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
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X, y)
            outputs['loss'].backward()
            self.optimizer.step()

            if self.logger is not None:
                self.logger.log({'train_loss': outputs['loss'].item()})

    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds, labels = [], []
        for X, y in test_loader:
            pred = self.model.predict(X.to(self.device)).detach().cpu().numpy()
            preds.append(pred)
            labels.append(y)
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        evaluation_result = {}
        for metric, fun in self.eval_funs.items():
            evaluation_result[metric] = fun(preds, labels)
        return evaluation_result

    def fit(
        self,
        train_X: Union[np.ndarray, torch.Tensor],
        train_y: Union[np.ndarray, torch.Tensor],
        valid_X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        valid_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        test_X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        test_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:

        train_X = torch.tensor(train_X) if not isinstance(train_X, torch.Tensor) else train_X
        train_y = torch.tensor(train_y) if not isinstance(train_y, torch.Tensor) else train_y
        train_loader = DataLoader(
            TensorDataset(train_X, train_y), 
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        if valid_X is not None and valid_y is not None:
            valid_X = torch.tensor(valid_X) if not isinstance(valid_X, torch.Tensor) else valid_X
            valid_y = torch.tensor(valid_y) if not isinstance(valid_y, torch.Tensor) else valid_y
            valid_loader = DataLoader(
                TensorDataset(valid_X, valid_y), 
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            valid_loader = None

        if test_X is not None and test_y is not None:
            test_X = torch.tensor(test_X) if not isinstance(test_X, torch.Tensor) else test_X
            test_y = torch.tensor(test_y) if not isinstance(test_y, torch.Tensor) else test_y
            test_loader = DataLoader(
                TensorDataset(test_X, test_y), 
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
            )
        else:
            test_loader = None

        best_valid_epoch = 0
        best_valid_score = float('inf') if self.metric == 'rmse' else -float('inf')
        selected_test_score = float('inf') if self.metric == 'rmse' else -float('inf')

        self.model.to(self.device)
        pbar = trange(1, self.n_epoch + 1)
        for curr_epoch in pbar:
            self.train(train_loader)

            all_results = {'epoch': curr_epoch}

            if valid_loader is not None:
                results = self.test(valid_loader)
                valid_results = {f"valid_{metric}": value for metric, value in results.items()}

                score = valid_results[f"valid_{self.metric}"]
                if self.metric == 'rmse':
                    self.early_stopping(score, 'min')
                    if score < best_valid_score:
                        best_valid_score = score
                        best_valid_epoch = curr_epoch
                        
                else:
                    self.early_stopping(score, 'max')
                    if score > best_valid_score:
                        best_valid_score = score
                        best_valid_epoch = curr_epoch

                valid_results |= {
                    'best_valid_epoch': best_valid_epoch,
                    'best_valid_score': best_valid_score,
                    'early_stop_counter': self.early_stopping.counter,
                }
                all_results |= valid_results

            if test_loader is not None:
                results = self.test(test_loader)
                test_results = {f"test_{metric}": value for metric, value in results.items()}

                if curr_epoch == best_valid_epoch:
                    selected_test_score = test_results[f"test_{self.metric}"]

                test_results |= {'selected_test_score': selected_test_score}
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
