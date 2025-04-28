from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TabularDatasetOutput:
    X: np.ndarray
    y: np.ndarray
    quantile: np.ndarray = None

    def __len__(self):
        return len(self.X)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, quantile: np.ndarray = None) -> None:
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        self.quantile = torch.Tensor(quantile) if quantile is not None else None

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i):
        return TabularDatasetOutput(
            X=self.X[i],
            y=self.y[i],
            quantile=self.quantile[i] if self.quantile is not None else None,
        )
    

def collate_fn(batch_data):
    keys = vars(batch_data[0]).keys()
    filtered_keys = [k for k in keys if getattr(batch_data[0], k) is not None]
    batch_dict = {k: torch.stack([getattr(x, k) for x in batch_data]) for k in filtered_keys}
    return TabularDatasetOutput(**batch_dict)
