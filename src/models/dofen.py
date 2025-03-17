from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .base import BaseClassifier, BaseRegressor


class Reshape(nn.Module):
    def __init__(self, *args: int) -> None:
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)


class FastGroupConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        self.fast_mode = kwargs.pop('fast_mode')
        nn.Conv1d.__init__(self, *args, **kwargs)
        if self.groups > self.fast_mode:
            self.weight = nn.Parameter(
                self.weight.reshape(
                    self.groups, self.out_channels // self.groups, self.in_channels // self.groups, 1
                ).permute(3, 0, 2, 1)
            )
            self.bias = nn.Parameter(self.bias.unsqueeze(0).unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.groups > self.fast_mode:
            x = x.reshape(-1, self.groups, self.in_channels // self.groups, 1)
            return (x * self.weight).sum(2, keepdims=True).permute(0, 1, 3, 2).reshape(-1, self.out_channels, 1) + self.bias
        else:
            return self._conv_forward(x, self.weight, self.bias)


class ConditionGeneration(nn.Module):
    def __init__(
        self,
        category_column_count: List[int],
        n_cond: int = 128,
        categorical_optimized: bool = False,
        fast_mode: int = 64,
    ) -> None:

        super(ConditionGeneration, self).__init__()
        self.fast_mode = fast_mode
        self.categorical_optimized = categorical_optimized

        index_info = self.extract_feature_metadata(category_column_count)
        self.numerical_index = index_info['numerical_index']
        self.categorical_index = index_info['categorical_index']
        self.categorical_count = index_info['categorical_count']

        categorical_offset = torch.tensor([0] + np.cumsum(self.categorical_count).tolist()[:-1]).long()
        self.register_buffer('categorical_offset', categorical_offset)

        self.n_cond = n_cond
        self.phi_1 = self.get_phi_1()

    def extract_feature_metadata(self, category_column_count: List[int]) -> Dict[str, List[int]]:
        numerical_index = [i for i, count in enumerate(category_column_count) if count == -1]
        categorical_index = [i for i, count in enumerate(category_column_count) if count != -1]
        categorical_count = [count for count in category_column_count if count != -1]
        return {
            'numerical_index': numerical_index,
            'categorical_index': categorical_index,
            'categorical_count': categorical_count,
        }

    def get_phi_1(self) -> nn.ModuleDict:
        phi_1 = nn.ModuleDict()
        if len(self.numerical_index):
            phi_1['num'] = nn.Sequential(
                # input = (b, n_num_col)
                # output = (b, n_num_col, n_cond)
                Reshape(-1, len(self.numerical_index), 1),
                FastGroupConv1d(
                    len(self.numerical_index),
                    len(self.numerical_index) * self.n_cond,
                    kernel_size=1,
                    groups=len(self.numerical_index),
                    fast_mode=self.fast_mode,
                ),
                nn.Sigmoid(),
                Reshape(-1, len(self.numerical_index), self.n_cond),
            )

        if len(self.categorical_index):
            phi_1['cat'] = nn.ModuleDict()
            phi_1['cat']['embedder'] = nn.Embedding(sum(self.categorical_count), self.n_cond)          
            phi_1['cat']['mapper'] = nn.Sequential(
                # input = (b, n_cat_col, n_cond)
                # output = (b, n_cat_col, n_cond)
                Reshape(-1, len(self.categorical_index) * self.n_cond, 1),
                nn.GroupNorm(len(self.categorical_index), len(self.categorical_index) * self.n_cond),
                FastGroupConv1d(
                    len(self.categorical_index) * self.n_cond,
                    len(self.categorical_index) * self.n_cond,
                    kernel_size=1,
                    groups=len(self.categorical_index) * self.n_cond if self.categorical_optimized else len(self.categorical_index),
                    fast_mode=self.fast_mode,
                ),                
                nn.Sigmoid(),
                Reshape(-1, len(self.categorical_index), self.n_cond)
            )
        return phi_1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = []

        if len(self.numerical_index):
            num_x = x[:, self.numerical_index].float()
            num_sample_emb = self.phi_1['num'](num_x)
            M.append(num_sample_emb)

        if len(self.categorical_index):
            cat_x = x[:, self.categorical_index].long() + self.categorical_offset
            cat_sample_emb = self.phi_1['cat']['mapper'](self.phi_1['cat']['embedder'](cat_x))
            M.append(cat_sample_emb)

        M = torch.cat(M, dim=1) # (b, n_col, n_cond)
        M = M.permute(0, 2, 1) # (b, n_cond, n_col)
        return M


class rODTConstruction(nn.Module):
    def __init__(self, n_cond: int, n_col: int) -> None:
        super().__init__()
        self.permutator = torch.rand(n_cond * n_col).argsort(-1)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        return M.reshape(M.shape[0], -1, 1)[:, self.permutator, :]


class rODTForestConstruction(nn.Module):
    def __init__(
        self,
        n_col: int,
        n_rodt: int,
        n_cond: int,
        n_estimator: int,
        n_head: int = 1,
        n_hidden: int = 128,
        n_forest: int = 100,
        dropout: float = 0.0,
        fast_mode: int = 64,
        device = torch.device('cuda'),
    ) -> None:

        super().__init__()
        self.device = device
        self.n_estimator = n_estimator
        self.n_forest = n_forest
        self.n_rodt = n_rodt
        self.n_head = n_head
        self.n_hidden = n_hidden

        self.phi_2 = nn.Sequential(
            nn.GroupNorm(n_rodt, n_cond * n_col),
            nn.Dropout(dropout),
            FastGroupConv1d(n_cond * n_col, n_cond * n_col, groups=n_rodt, kernel_size=1, fast_mode=fast_mode),
            nn.ReLU(),
            nn.GroupNorm(n_rodt, n_cond * n_col),
            nn.Dropout(dropout),
            FastGroupConv1d(n_cond * n_col, n_rodt * n_head, groups=n_rodt, kernel_size=1, fast_mode=fast_mode), 
            Reshape(-1, n_rodt, n_head)
        )
        self.E = nn.Embedding(n_rodt, n_hidden)
        self.sample_without_replacement_eval = self.get_sample_without_replacement()

    def get_sample_without_replacement(self) -> torch.Tensor:
        return torch.rand(self.n_forest, self.n_rodt, device=self.device).argsort(-1)[:, :self.n_estimator]

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        b = O.shape[0]
        w = self.phi_2(O) # (b, n_rodt, n_head)
        E = self.E.weight.unsqueeze(0) # (1, n_rodt, n_hidden)

        sample_without_replacement = self.get_sample_without_replacement() if self.training else self.sample_without_replacement_eval

        w_prime = w[:, sample_without_replacement].softmax(-2).unsqueeze(-1) # (b, n_forest, n_rodt, n_head, 1)
        E_prime = E[:, sample_without_replacement].reshape(
            1, self.n_forest, self.n_estimator, self.n_head, self.n_hidden // self.n_head
        ) # (1, n_forest, n_rodt, n_head, n_hidden // n_head)
        F = (w_prime * E_prime).sum(-3).reshape(
            b, self.n_forest, self.n_hidden
        ) # (b, n_forest, n_hidden)
        return F


class rODTForestBagging(nn.Module):
    def __init__(self, n_hidden: int, dropout: float, n_class: int) -> None:
        super().__init__()
        self.phi_3 = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.LayerNorm(n_hidden),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        return self.phi_3(F) # (b, n_forest, n_class)


class DOFEN(nn.Module):
    def __init__(
        self,
        category_column_count: List[int],
        n_class: int,
        m: int = 16,
        d: int = 4,
        n_head: int = 1,
        n_forest: int = 100,
        n_hidden: int = 128,
        dropout: float = 0.0,
        categorical_optimized: bool = False,
        fast_mode: int = 2048,
        use_bagging_loss: bool = False,
        device = torch.device('cuda'),
        verbose: bool = False,
    ):
        super().__init__()

        self.device = device
        self.n_class = 1 if n_class == -1 else n_class
        self.is_rgr = True if n_class == -1 else False

        self.m = m
        self.d = d
        self.n_head = n_head
        self.n_forest = n_forest
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.use_bagging_loss = use_bagging_loss

        self.n_cond = self.d * self.m
        self.n_col = len(category_column_count)
        self.n_rodt = self.n_cond * self.n_col // self.d
        self.n_estimator = max(2, int(self.n_col ** 0.5)) * self.n_cond // self.d

        self.condition_generation = ConditionGeneration(            
            category_column_count,
            n_cond=self.n_cond,
            categorical_optimized=categorical_optimized,
            fast_mode=fast_mode,
        )
        self.rodt_construction = rODTConstruction(
            self.n_cond,
            self.n_col,
        )
        self.rodt_forest_construction = rODTForestConstruction(
            self.n_col,
            self.n_rodt,
            self.n_cond,
            self.n_estimator,
            n_head=self.n_head,
            n_hidden=self.n_hidden,
            n_forest=self.n_forest,
            dropout=self.dropout,
            fast_mode=fast_mode,
            device=self.device,
        )
        self.rodt_forest_bagging = rODTForestBagging(
            self.n_hidden,
            self.dropout,
            self.n_class,
        )

        if verbose:
            print('='*20)
            print('total condition: ', self.n_cond * self.n_col)
            print('n_rodt: ', self.n_rodt)
            print('n_estimator: ', self.n_estimator)    
            print('='*20)

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        M = self.condition_generation(x) # (b, n_cond, n_col)
        O = self.rodt_construction(M) # (b, n_rodt, d)
        F = self.rodt_forest_construction(O) # (b, n_forest, n_hidden)
        y_hats = self.rodt_forest_bagging(F) # (b, n_forest, n_class)
        y_hat = y_hats.mean(1) # (b, n_class)

        if y is not None:
            loss = self.compute_loss(
                y_hats.permute(0, 2, 1) if not self.is_rgr else y_hats, 
                y.unsqueeze(-1).expand(-1, self.n_forest)
            )
            if self.n_forest > 1 and self.training and self.use_bagging_loss:
                loss += self.compute_loss(y_hat, y)
            return {'pred': y_hat, 'loss': loss}
        return {'pred': y_hat}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['pred']


class DOFENClassifier(BaseClassifier, DOFEN):
    def __init__(
        self,
        category_column_count: List[int],
        n_class: int,
        m: int = 16,
        d: int = 4,
        n_head: int = 4,
        n_forest: int = 100,
        n_hidden: int = 128,
        dropout: float = 0.0,
        categorical_optimized: bool = False,
        fast_mode: int = 2048,
        use_bagging_loss: bool = False,
        device = torch.device('cuda'),
        verbose: bool = False,
    ) -> None:

        super().__init__(
            category_column_count=category_column_count,
            n_class=n_class,
            m=m,
            d=d,
            n_head=n_head,
            n_forest=n_forest,
            n_hidden=n_hidden,
            dropout=dropout,
            categorical_optimized=categorical_optimized,
            fast_mode=fast_mode,
            use_bagging_loss=use_bagging_loss,
            device=device,
            verbose=verbose,
        )


class DOFENRegressor(BaseRegressor, DOFEN):
    def __init__(
        self,
        category_column_count: List[int],
        m: int = 16,
        d: int = 4,
        n_head: int = 4,
        n_forest: int = 100,
        n_hidden: int = 128,
        dropout: float = 0.0,
        categorical_optimized: bool = False,
        fast_mode: int = 2048,
        use_bagging_loss: bool = False,
        device=torch.device('cuda'),
        verbose: bool = False,
    ) -> None:

        super().__init__(
            category_column_count=category_column_count,
            n_class=-1,
            m=m,
            d=d,
            n_head=n_head,
            n_forest=n_forest,
            n_hidden=n_hidden,
            dropout=dropout, 
            categorical_optimized=categorical_optimized,
            fast_mode=fast_mode,
            use_bagging_loss=use_bagging_loss,
            device=device,
            verbose=verbose,
        )
