from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ReGLU(nn.Module):
    """
    Rectified Gated Linear Unit (ReGLU)
    reference: https://arxiv.org/abs/2002.05202
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.relu(x2)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU)
    reference: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        mid_features = int(in_features * 8 / 3)
        self.WG = nn.Linear(in_features, mid_features, bias=False)
        self.W1 = nn.Linear(in_features, mid_features, bias=False)
        self.W2 = nn.Linear(mid_features, out_features, bias=False)

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W2(self.swish(self.WG(x)) * self.W1(x))


class NumericalEmbeddingLayer(nn.Module):
    def __init__(self, num: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, d))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
    
    def size(self) -> torch.Size:
        return self.weight.size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[None] * x[:, :, None]


class BiasLayer(nn.Module):
    def __init__(self, num: int, d: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.empty(num, d))
        nn.init.kaiming_uniform_(self.bias, a=5 ** 0.5)
    
    def size(self) -> torch.Size:
        return self.bias.size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class ResidualLayer(nn.Module):
    def __init__(self, d: int, residual_dropout_rate: float) -> None:
        super().__init__()
        self.residual_dropout_rate = residual_dropout_rate
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:
        if self.residual_dropout_rate > 0:
            x_res = F.dropout(x_res, p=self.residual_dropout_rate, training=self.training)
        return self.norm(x + x_res)

class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        category_column_count: List[int],
        d: int, # embedding dimension
        use_bias: bool = True,
    ):
        super().__init__()
        self.d = d
        self.use_bias = use_bias

        index_info = self.extract_feature_metadata(category_column_count)
        self.numerical_index = index_info['numerical_index']
        self.categorical_index = index_info['categorical_index']
        self.categorical_count = index_info['categorical_count']

        categorical_offset = torch.tensor([0] + np.cumsum(self.categorical_count).tolist()[:-1]).long()
        self.register_buffer('categorical_offset', categorical_offset)

        self.num_feat = len(self.numerical_index) + len(self.categorical_index)
        self.num_tokens = self.num_feat + 1 # with [CLS] token
        self.layers = self.create_layers()

    def extract_feature_metadata(self, category_column_count: List[int]) -> Dict[str, List[int]]:
        numerical_index = [i for i, count in enumerate(category_column_count) if count == -1]
        categorical_index = [i for i, count in enumerate(category_column_count) if count != -1]
        categorical_count = [count for count in category_column_count if count != -1]
        return {
            'numerical_index': numerical_index,
            'categorical_index': categorical_index,
            'categorical_count': categorical_count,
        }

    def create_layers(self) -> nn.ModuleDict:
        layers = nn.ModuleDict()
        layers['numerical'] = NumericalEmbeddingLayer(len(self.numerical_index) + 1, self.d) # add [CLS] token layer
        if self.categorical_index:
            layers['categorical'] = nn.Embedding(sum(self.categorical_count), self.d)
            nn.init.kaiming_uniform_(layers['categorical'].weight, a=5 ** 0.5)
        if self.use_bias:
            layers['bias'] = BiasLayer(self.num_feat, self.d)
        return layers

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        x_numerical = x[:, self.numerical_index] if self.numerical_index else None
        x_categorical = x[:, self.categorical_index] if self.categorical_index else None
        x = torch.ones(x.size(0), 1, device=x.device) # add [CLS] token

        # Process numerical features
        if self.numerical_index:
            x = torch.cat([x, x_numerical], dim=1) # shape: (batch_size, num_feat + 1)
        x = self.layers['numerical'](x) # shape: (batch_size, num_feat + 1, d)

        # Process categorical features
        if self.categorical_index:
            x_categorical = x_categorical.long() + self.categorical_offset # shape: (batch_size, cat_feat)
            x_categorical = self.layers['categorical'](x_categorical) # shape: (batch_size, cat_feat, d)
            x = torch.cat([x, x_categorical], dim=1) # shape: (batch_size, num_feat + cat_feat + 1, d)

        # Add bias to non-[CLS] tokens
        if self.use_bias:
            x[:, 1:] = self.layers['bias'](x[:, 1:])
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = self.create_linear(embed_dim, embed_dim) # query
        self.k_proj = self.create_linear(embed_dim, embed_dim) # key
        self.v_proj = self.create_linear(embed_dim, embed_dim) # value
        self.out_proj = self.create_linear(embed_dim, embed_dim) # output
        self.dropout = nn.Dropout(dropout)

    def create_linear(self, in_features: int, out_features: int):
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
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query)  # shape: (batch_size, tgt_len, embed_dim)
        k = self.k_proj(key)    # shape: (batch_size, src_len, embed_dim)
        v = self.v_proj(value)  # shape: (batch_size, src_len, embed_dim)

        # Reshape into multihead format
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, tgt_len, head_dim)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2) # shape: (batch_size, num_heads, src_len, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # shape: (batch_size, num_heads, tgt_len, src_len)
        attn_scores = attn_scores / self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1) # shape: (batch_size, num_heads, tgt_len, src_len)
        attn_weights = self.dropout(attn_weights) 
        attn_output = torch.matmul(attn_weights, v) # shape: (batch_size, num_heads, tgt_len, head_dim)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim) # shape: (batch_size, tgt_len, embed_dim)
        output = self.out_proj(attn_output)
        return output


class MultiheadDiffAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0., depth: int = 1, use_rms_norm: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        self.depth = depth
        self.use_rms_norm = use_rms_norm
        # assert self.head_dim * num_heads * 2 == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = self.create_linear(embed_dim, embed_dim*2) # query
        self.k_proj = self.create_linear(embed_dim, embed_dim*2) # key
        self.v_proj = self.create_linear(embed_dim, embed_dim) # value
        self.out_proj = self.create_linear(embed_dim, embed_dim) # output
        self.dropout = nn.Dropout(dropout)

        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        #self.norm = nn.RMSNorm(self.head_dim * 2, eps=1e-5, elementwise_affine=True)
        self.norm = nn.GroupNorm(num_heads, embed_dim)

    @staticmethod
    def create_linear(in_features: int, out_features: int) -> nn.Linear:
        linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(linear.weight, gain=1 / 2 ** 0.5)
        return linear

    @staticmethod
    def lambda_init_fn(depth: int) -> torch.Tensor:
        return 0.8 - 0.6 * torch.exp(torch.tensor(-0.3 * depth))

    def compute_lambda(self) -> torch.Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        query: Tensor of shape (batch_size, tgt_len, embed_dim)
        key: Tensor of shape (batch_size, src_len, embed_dim)
        value: Tensor of shape (batch_size, src_len, embed_dim)
        attn_mask: Optional[Tensor] of shape (tgt_len, src_len) or (batch_size, tgt_len, src_len)
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query)  # shape: (batch_size, tgt_len, embed_dim)
        k = self.k_proj(key)    # shape: (batch_size, src_len, embed_dim)
        v = self.v_proj(value)  # shape: (batch_size, src_len, embed_dim)

        # Reshape into multihead format
        q = q.view(batch_size, tgt_len, self.num_heads * 2, self.head_dim * 2).transpose(1, 2) # shape: (batch_size, num_heads * 2, tgt_len, head_dim)
        k = k.view(batch_size, src_len, self.num_heads * 2, self.head_dim * 2).transpose(1, 2) # shape: (batch_size, num_heads * 2, src_len, head_dim)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim * 2).transpose(1, 2) # shape: (batch_size, num_heads, src_len, head_dim * 2)

        lam = self.compute_lambda()
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # shape: (batch_size, num_heads * 2, tgt_len, src_len)
        attn_scores = attn_scores / self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1) # shape: (batch_size, num_heads * 2, tgt_len, src_len)
        attn_weights = attn_weights.view(batch_size, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lam * attn_weights[:, :, 1]
        attn_weights = self.dropout(attn_weights)         
        attn_output = torch.matmul(attn_weights, v) # shape: (batch_size, num_heads, tgt_len, head_dim)

        if self.use_rms_norm:
            attn_output = self.norm(
                attn_output \
                    .permute(0, 2, 1, 3) \
                    .contiguous() \
                    .flatten(2) \
                    .permute(0, 2, 1) \
            ).permute(0, 2, 1) \
            .reshape(batch_size, tgt_len, self.num_heads, -1) \
            .permute(0, 2, 1, 3) \
            .contiguous()
            attn_output *= (1 - self.lambda_init)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim) # shape: (batch_size, tgt_len, embed_dim)
        output = self.out_proj(attn_output)
        return output
