import torch
import torch.nn.functional as F
from torch import nn


class BaseClassifier(nn.Module):
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y.long() if y.dtype != torch.long else y
            return torch.nn.functional.cross_entropy(y_hat, y)

    def predict(self, X: torch.Tensor, *arg, **kwargs) -> torch.Tensor:
        return super().predict(X, *arg, **kwargs).argmax(dim=1)


class BaseRegressor(nn.Module):
     def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float() if y.dtype != torch.float else y
        y = y if y.shape[-1] == 1 else y.unsqueeze(-1)
        return F.mse_loss(y_hat, y)
