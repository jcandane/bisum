import torch

import pytest

from bisum import dense_intra

@pytest.fixture(scope="module")
def A():
    return torch.rand((3,3,3,3,3,3,3,3))

@pytest.fixture(scope="module")
def iqwe():
    return torch.tensor([67, 84, 101, 101, 84, 85, 84, 84])

## TODO assert something substantial

def test_den_tensor_intraTr(A, iqwe):
    X = dense_intra.den_tensor_intraTr(A, iqwe, torch.tensor([]))
    assert X.shape == (3, 3, 3, 3)
