import random
from pathlib import Path

import numpy as np
import torch

from utils import load_pkl_data, set_random_seed


ROOT = '/home/jiawei/Desktop/github/DOFEN'


def test_load_pkl_data():
    data_ids = [
        361055,
        361060,
        361061,
        361062,
        361065,
        361069,
        361068,
        361274,
        361276,
        361113,
        361072,
        361073,
        361074,
        361076,
        361077,
        361276,
        361097,
        361287,
    ]

    expected_keys = {
        'x_train',
        'y_train',
        'x_val',
        'y_val',
        'x_test',
        'y_test',
        'label_cat_count',
        'col_cat_count',
    }

    for data_id in data_ids:
        loaded_data = load_pkl_data(Path(ROOT, 'tabular-benchmark/tabular_benchmark_data', str(data_id), '0.pkl'))
        assert expected_keys <= loaded_data.keys(), 'Loaded data does not contain the expected keys'


def test_set_random_seed():
    random_seed = 42

    # Set random seed
    set_random_seed(random_seed)

    # Test NumPy random consistency
    np_result_1 = np.random.rand(3)
    set_random_seed(random_seed)
    np_result_2 = np.random.rand(3)
    assert np.array_equal(np_result_1, np_result_2), 'NumPy random numbers are not consistent'

    # Test Python random consistency
    random_result_1 = [random.randint(0, 100) for _ in range(5)]
    set_random_seed(random_seed)
    random_result_2 = [random.randint(0, 100) for _ in range(5)]
    assert random_result_1 == random_result_2, 'Python random numbers are not consistent'

    # Test PyTorch random consistency
    torch_result_1 = torch.rand(3)
    set_random_seed(random_seed)
    torch_result_2 = torch.rand(3)
    assert torch.equal(torch_result_1, torch_result_2), 'PyTorch random numbers are not consistent'

    # If GPU is available, test CUDA random consistency
    if torch.cuda.is_available():
        torch_result_cuda_1 = torch.rand(3, device='cuda')
        set_random_seed(random_seed)
        torch_result_cuda_2 = torch.rand(3, device='cuda')
        assert torch.equal(torch_result_cuda_1, torch_result_cuda_2), 'CUDA random numbers are not consistent'
