from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .layers import ReGLU, ResidualLayer, FeatureTokenizer, MultiheadAttention


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_class,
        category_column_count: List[int],
        d_token: int = 192,
        d_ffn_factor: float = 4 / 3,
        n_layer: int = 3,
        n_head: int = 8,
        attention_dropout_rate: float = 0.2,
        ffn_dropout_rate: float = 0.1,
        residual_dropout_rate: float = 0.,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        self.n_class = n_class
        self.ffn_dropout_rate = ffn_dropout_rate

        self.feat_tokenizer = FeatureTokenizer(
            category_column_count=category_column_count,
            d=d_token,
            use_bias=use_bias,
        )
        self.num_tokens = self.feat_tokenizer.num_tokens

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for _ in range(n_layer):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(d_token, n_head, attention_dropout_rate),
                    'linear0': nn.Linear(d_token, d_hidden * 2),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'res0': ResidualLayer(d_token, residual_dropout_rate),
                    'res1': ResidualLayer(d_token, residual_dropout_rate),
                }
            )
            self.layers.append(layer)

        self.act = ReGLU()
        self.last_act = nn.ReLU()
        self.head = nn.Linear(d_token, n_class)

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def evaluate(self, X, y):
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ):
        x = self.feat_tokenizer(x)
        x_res = x
        for i, layer in enumerate(self.layers):
            if i != len(self.layers) - 1: 
                x_res = layer['attention'](x_res, x_res, x_res)
            else: # last layer only process [CLS] token
                x_res = layer['attention'](x_res[:, :1], x_res, x_res)

            x = layer['res0'](x[:, :x_res.shape[1]], x_res) # residual layer

            x_res = x
            x_res = self.act(layer['linear0'](x_res))
            if self.ffn_dropout_rate > 0:
                x_res = F.dropout(x_res, p=self.ffn_dropout_rate, training=self.training)
            x_res = layer['linear1'](x_res)
            x = layer['res1'](x, x_res)

        y_hat = self.head(self.last_act(x[:, 0]))

        if y is not None:
            loss = self.compute_loss(y_hat, y)
            return {'pred': y_hat, 'loss': loss}
        return {'pred': y_hat}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['pred']


class FTTransformerClassifier(FTTransformer):
    def __init__(
        self,
        n_class,
        category_column_count: List[int],
        d_token: int = 192,
        d_ffn_factor: float = 4 / 3,
        n_layer: int = 3,
        n_head: int = 8,
        attention_dropout_rate: float = 0.2,
        ffn_dropout_rate: float = 0.1,
        residual_dropout_rate: float = 0.,
        use_bias: bool = True,
    ) -> None:

        super().__init__(
            n_class=n_class,
            category_column_count=category_column_count,
            d_token=d_token,
            d_ffn_factor=d_ffn_factor,
            n_layer=n_layer,
            n_head=n_head,
            attention_dropout_rate=attention_dropout_rate,
            ffn_dropout_rate=ffn_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            use_bias=use_bias,
        )
    
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.long() if y.dtype != torch.long else y
        return torch.nn.functional.cross_entropy(y_hat, y)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return super().predict(x).argmax(dim=1)


class FTTransformerRegressor(FTTransformer):
    def __init__(
        self,
        category_column_count: List[int],
        d_token: int = 192,
        d_ffn_factor: float = 4 / 3,
        n_layer: int = 3,
        n_head: int = 8,
        attention_dropout_rate: float = 0.2,
        ffn_dropout_rate: float = 0.1,
        residual_dropout_rate: float = 0.,
        use_bias: bool = True,
    ) -> None:

        super().__init__(
            n_class=1,
            category_column_count=category_column_count,
            d_token=d_token,
            d_ffn_factor=d_ffn_factor,
            n_layer=n_layer,
            n_head=n_head,
            attention_dropout_rate=attention_dropout_rate,
            ffn_dropout_rate=ffn_dropout_rate,
            residual_dropout_rate=residual_dropout_rate,
            use_bias=use_bias,
        )

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float() if y.dtype != torch.float else y
        return torch.nn.functional.mse_loss(y_hat, y)
