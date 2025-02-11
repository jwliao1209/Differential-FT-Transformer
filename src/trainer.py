from typing import Dict, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        n_epoch: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        eval_funs: Optional[Dict[str, callable]] = None,
        metric: str = 'accuracy',
        logger: Optional[object] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> None:

        self.model = model
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.eval_funs = eval_funs
        self.metric = metric
        self.logger = logger
        self.device = torch.device(device)
        self.set_optimization()

    def set_optimization(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
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

        best_score = -float('inf')
        best_epoch = 0

        self.model.to(self.device)
        for curr_ep in trange(1, self.n_epoch + 1):
            self.train(train_loader)
            if test_loader is not None:
                evaluation_result = self.test(test_loader)

                if evaluation_result[self.metric] > best_score:
                    best_score = evaluation_result[self.metric]
                    best_epoch = curr_ep
                
                results = {'epoch': curr_ep} | evaluation_result | {'best_epoch': best_epoch, 'best_score': best_score}
                tqdm.write(str(results))

                if self.logger is not None:
                    self.logger.log(results)

        self.model.cpu()
