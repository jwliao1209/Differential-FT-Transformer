import pickle
import random
from typing import Dict

import numpy as np
import torch


def load_pkl_data(path: str) -> Dict[str, np.ndarray]:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def set_random_seed(random_seed: int = 0, deterministic: bool = True) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
