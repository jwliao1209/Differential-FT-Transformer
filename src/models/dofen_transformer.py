import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseClassifier, BaseRegressor
from .dofen import Reshape, FastGroupConv1d


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_scale: int = 1000.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_scale = max_scale

        inv_freq = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(max_scale) / d_model)
        )
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, quantile_X: torch.Tensor) -> torch.Tensor:
        """
        quantile_X: Tensor, shape = (batch, num_features), values in [0, 1]
        Returns:
            sinusoidal embedding: shape = (batch, num_features, d_model)
        """
        B, F = quantile_X.shape
        pos = quantile_X.unsqueeze(-1)  # (B, F, 1)
        angle = pos * self.inv_freq  # (B, F, d_model//2)
        emb = torch.zeros(B, F, self.d_model, device=quantile_X.device)
        emb[..., 0::2] = torch.sin(angle)
        emb[..., 1::2] = torch.cos(angle)
        return emb.unsqueeze(2)


class ConditionGeneration(nn.Module):
    def __init__(
        self,
        category_column_count: List[int],
        n_cond: int = 128,
        n_hidden: int = 4,
        categorical_optimized: bool = False,
        fast_mode: int = 64,
        use_quantile_emb: bool = True,
    ) -> None:

        super(ConditionGeneration, self).__init__()
        self.n_cond = n_cond
        self.n_hidden = n_hidden
        self.use_quantile_emb = use_quantile_emb
        self.fast_mode = fast_mode
        self.categorical_optimized = categorical_optimized

        index_info = self.extract_feature_metadata(category_column_count)
        self.numerical_index = index_info['numerical_index']
        self.categorical_index = index_info['categorical_index']
        self.categorical_count = index_info['categorical_count']

        categorical_offset = torch.tensor([0] + np.cumsum(self.categorical_count).tolist()[:-1]).long()
        self.register_buffer('categorical_offset', categorical_offset)

        self.pos_emb = SinusoidalEmbedding(self.n_hidden, max_scale=1000.0)
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
                # output = (b, n_num_col, n_cond, n_hidden)
                Reshape(-1, len(self.numerical_index), 1),
                nn.Linear(1, self.n_cond * self.n_hidden),
                # (b, n_num_col, 1) -> (b, n_num_col, n_cond * n_hidden)
                Reshape(-1, len(self.numerical_index), self.n_cond, self.n_hidden)
                # (b, n_num_col, n_cond * n_hidden) -> (b, n_num_col, n_cond, n_hidden)
            )

            # phi_1['num'] = nn.Sequential(
            #     # input = (b, n_num_col)
            #     # output = (b, n_num_col, n_cond)
            #     Reshape(-1, len(self.numerical_index), 1),
            #     FastGroupConv1d(
            #         len(self.numerical_index),
            #         len(self.numerical_index) * self.n_cond * self.n_hidden,
            #         kernel_size=1,
            #         groups=len(self.numerical_index),
            #         fast_mode=self.fast_mode,
            #     ),
            #     nn.Sigmoid(),
            #     Reshape(-1, len(self.numerical_index), self.n_cond, self.n_hidden),
            # )

            phi_1['num_comb'] = nn.Linear(2 * self.n_hidden, self.n_hidden)

        if len(self.categorical_index):
            phi_1['cat'] = nn.ModuleDict()
            phi_1['cat']['embedder'] = nn.Embedding(sum(self.categorical_count), self.n_cond * self.n_hidden)            
            phi_1['cat']['mapper'] = nn.Sequential(
                # input = (b, n_cat_col, n_cond)
                # output = (b, n_cat_col, n_cond)
                Reshape(-1, len(self.categorical_index) * self.n_cond  * self.n_hidden, 1),
                nn.GroupNorm(len(self.categorical_index), len(self.categorical_index) * self.n_cond  * self.n_hidden),
                FastGroupConv1d(
                    len(self.categorical_index) * self.n_cond  * self.n_hidden,
                    len(self.categorical_index) * self.n_cond  * self.n_hidden,
                    kernel_size=1,
                    groups=len(self.categorical_index) * self.n_cond  * self.n_hidden if self.categorical_optimized else len(self.categorical_index),
                    fast_mode=self.fast_mode),                
                nn.Sigmoid(),
                Reshape(-1, len(self.categorical_index), self.n_cond, self.n_hidden)
            )
        return phi_1

    def forward(self, x: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
        M = []

        if len(self.numerical_index):
            num_x = x[:, self.numerical_index].float()
            num_sample_emb = self.phi_1['num'](num_x)

            if self.use_quantile_emb:
                num_stats_emb = self.pos_emb(quantile)
                num_stats_emb_expand = num_stats_emb.expand(num_sample_emb.shape)
                M.append(
                    self.phi_1['num_comb'](torch.cat([num_sample_emb, num_stats_emb_expand], dim=-1))
                )
            else:
                M.append(num_sample_emb)

        if len(self.categorical_index):
            cat_x = x[:, self.categorical_index].long() + self.cat_offset
            cat_sample_emb = self.phi_1['cat']['mapper'](self.phi_1['cat']['embedder'](cat_x))
            M.append(cat_sample_emb)

        M = torch.cat(M, dim=1) # (b, n_col, n_cond, n_hidden)
        M = M.permute(0, 2, 1, 3) # (b, n_cond, n_col, n_hidden)
        return M


class rODTConstruction(nn.Module):
    def __init__(self, n_cond: int, n_col: int, d: int) -> None:
        super().__init__()
        self.permutator = torch.rand(n_cond * n_col).argsort(-1)
        self.d = d

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        b, _, _, embed_dim = M.shape
        return M.reshape(b, -1, embed_dim)[:, self.permutator, :].reshape(b, -1, self.d, embed_dim)


class rODTForestConstruction(nn.Module):
    def __init__(
        self,
        n_rodt: int,
        n_estimator: int,
        n_head: int = 1,
        n_hidden: int = 128,
        n_forest: int = 100,
        device = torch.device('cuda'),
    ) -> None:

        super().__init__()
        self.device = device
        self.n_estimator = n_estimator
        self.n_forest = n_forest
        self.n_rodt = n_rodt
        self.n_head = n_head
        self.n_hidden = n_hidden

        self.sample_without_replacement_eval = self.get_sample_without_replacement()

    def get_sample_without_replacement(self) -> torch.Tensor:
        return torch.rand(self.n_forest, self.n_rodt, device=self.device).argsort(-1)[:, :self.n_estimator]

    def forward(self, w, E) -> torch.Tensor:
        # w: (b, n_rodt, 1)
        # E: (b, n_rodt, n_hidden)
    
        sample_without_replacement = self.get_sample_without_replacement() if self.training else self.sample_without_replacement_eval

        w_prime = w[:, sample_without_replacement].softmax(-2) # (b, n_forest, n_rodt, 1)
        E_prime = E[:, sample_without_replacement].reshape(
            E.shape[0], self.n_forest, self.n_estimator, self.n_hidden
        ) # (b, n_forest, n_rodt, n_hidden)

        F = (w_prime * E_prime).sum(-2).reshape(
            E.shape[0], self.n_forest, self.n_hidden
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
            nn.Linear(n_hidden, n_class)
        )

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        return self.phi_3(F) # (b, n_forest, n_class)


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qw_proj = self.create_linear(embed_dim, embed_dim)
        self.kw_proj = self.create_linear(embed_dim, embed_dim)
        self.vw_proj = self.create_linear(embed_dim, embed_dim)
        self.ow_proj = self.create_linear(embed_dim, 1)

        self.qE_proj = self.create_linear(embed_dim, embed_dim)
        self.kE_proj = self.create_linear(embed_dim, embed_dim)
        self.vE_proj = self.create_linear(embed_dim, embed_dim)
        self.oE_proj = self.create_linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def create_linear(self, in_features: int, out_features: int) -> nn.Linear:
        linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(linear.weight, gain=1 / 2 ** 0.5)
        return linear

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        query: Tensor of shape (batch_size, tgt_len, embed_dim)
        key: Tensor of shape (batch_size, src_len, embed_dim)
        value: Tensor of shape (batch_size, src_len, embed_dim)
        attn_mask: Optional[Tensor] of shape (tgt_len, src_len) or (batch_size, tgt_len, src_len)
        """
        batch_size, tgt_len, _ = query.size()
        src_len = key.size(1)

        # Compute w
        qw = self.qw_proj(query)  # shape: (batch_size, tgt_len, embed_dim)
        kw = self.kw_proj(key)    # shape: (batch_size, src_len, embed_dim)
        vw = self.vw_proj(value)  # shape: (batch_size, src_len, 1)

        # Reshape into multihead format
        qw = qw.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, tgt_len, head_dim)
        kw = kw.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, head_dim)
        vw = vw.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, 1)

        attn_w = F.softmax(torch.matmul(qw, kw.transpose(-2, -1)) / self.scaling, dim=-1) # shape: (batch_size, num_heads, tgt_len, src_len)
        attn_w = self.dropout(attn_w) 
        attn_w_output = torch.matmul(attn_w, vw) # shape: (batch_size, num_heads, tgt_len, head_dim)

        # Reshape back to original dimensions
        attn_w_output = attn_w_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim) # shape: (batch_size, tgt_len, embed_dim)
        w_output = self.ow_proj(attn_w_output) # shape: (batch_size, tgt_len, 1)
        w_output = w_output.mean(1)

        # Compute w
        qE = self.qE_proj(query)  # shape: (batch_size, tgt_len, embed_dim)
        kE = self.kE_proj(key)    # shape: (batch_size, src_len, embed_dim)
        vE = self.vE_proj(value)  # shape: (batch_size, src_len, embed_dim)

        # Reshape into multihead format
        qE = qE.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, tgt_len, head_dim)
        kE = kE.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, head_dim)
        vE = vE.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, 1)

        attn_E = F.softmax(torch.matmul(qE, kE.transpose(-2, -1)) / self.scaling, dim=-1) # shape: (batch_size, num_heads, tgt_len, src_len)
        attn_E = self.dropout(attn_E) 
        attn_E_output = torch.matmul(attn_E, vE) # shape: (batch_size, num_heads, tgt_len, head_dim)

        # Reshape back to original dimensions
        attn_E_output = attn_E_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.embed_dim) # shape: (batch_size, tgt_len, embed_dim)
        E_output = self.oE_proj(attn_E_output) # shape: (batch_size, tgt_len, embed_dim)
        E_output = E_output.mean(1)

        return w_output, E_output


class DOFENTransformer(nn.Module):
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
        use_quantile_emb: bool = True,
        device: torch.device = torch.device('cuda'),
    ) -> None:

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
            n_hidden=self.n_hidden,
            categorical_optimized=categorical_optimized,
            fast_mode=fast_mode,
            use_quantile_emb=use_quantile_emb,
        )

        self.rodt_construction = rODTConstruction(
            self.n_cond,
            self.n_col,
            self.d,
        )

        self.norm = nn.LayerNorm(n_hidden)
        self.attn = MultiheadAttention(
            embed_dim=self.n_hidden,
            num_heads=n_head,
            dropout=0.2,
        )
        self.rodt_forest_construction = rODTForestConstruction(
            n_rodt=self.n_rodt, 
            n_estimator=self.n_estimator,
            n_head=self.n_head, 
            n_hidden=self.n_hidden,
            n_forest=self.n_forest,
            device=self.device
        )
        self.rodt_forest_bagging = rODTForestBagging(
            self.n_hidden,
            self.dropout,
            self.n_class
        )

        self.ffn_dropout_rate = 0.1

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        quantile: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> None:

        M = self.condition_generation(X, quantile) # (b, n_cond, n_col, n_hidden)
        temp = M.permute(0, 2, 1, 3).reshape(-1, self.n_cond, self.n_hidden) # (b * n_col, n_cond, n_hidden)

        # cos similarity loss
        temp_norm = temp / torch.norm(temp, dim=-1, keepdim=True)
        temp_matrix = torch.einsum('bij,bjk->bik', temp_norm, temp_norm.permute(0, 2, 1))
        sim_loss = temp_matrix.abs().mean()

        O = self.rodt_construction(M)              # (b, n_rodt, d, n_hidden)
        O = O.reshape(-1, self.d, self.n_hidden)   # (b * n_rodt, d, n_hidden)
        O = self.norm(O)
        w, E = self.attn(O, O, O)

        w = w.reshape(-1, self.n_rodt, 1)
        E = E.reshape(-1, self.n_rodt, self.n_hidden)

        # E_norm = E / E.norm(dim=-1, keepdim=True)
        # sim_loss = torch.einsum('bij,bjk->bik', E_norm, E_norm.permute(0, 2, 1))

        F = self.rodt_forest_construction(w, E) # (b, n_forest, n_hidden)
        y_hats = self.rodt_forest_bagging(F)    # (b, n_rodt, n_class)
        y_hat = y_hats.mean(1)                  # (b, n_class)

        if y is not None:
            loss = self.compute_loss(
                y_hats.permute(0, 2, 1) if not self.is_rgr else y_hats, 
                y.unsqueeze(-1).expand(-1, self.n_forest)
            ) #+ sim_loss
            if self.n_forest > 1 and self.training and self.use_bagging_loss:
                loss += self.compute_loss(y_hat, y)
            return {'pred': y_hat, 'loss': loss}
        return {'pred': y_hat}

    def predict(self, X: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
        return self.forward(X=X, quantile=quantile)['pred']


class DOFENTransformerClassifier(BaseClassifier, DOFENTransformer):
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
        use_quantile_emb: bool = True,
        device: torch.device = torch.device('cuda'),
        *args,
        **kwargs,
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
            use_quantile_emb=use_quantile_emb,
            device=device,
        )


class DOFENTransformerRegressor(BaseRegressor, DOFENTransformer):
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
        use_quantile_emb: bool = True,
        device: torch.device = torch.device('cuda'),
        *args,
        **kwargs,
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
            use_quantile_emb=use_quantile_emb,
            device=device,
        )
