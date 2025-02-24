import torch
from torch import nn


class BaseClassifier(nn.Module):
    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y = y.long() if y.dtype != torch.long else y
            return torch.nn.functional.cross_entropy(y_hat, y)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return super().predict(x).argmax(dim=1)


class BaseRegressor(nn.Module):
     def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.float() if y.dtype != torch.float else y
        y = y.unsqueeze(1) if y.ndim == 1 else y
        return torch.nn.functional.mse_loss(y_hat, y)
