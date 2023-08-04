import torch

import pytest

from bisum import labels

@pytest.fixture
def A():
    return torch.tensor([1,8,-5,3])

@pytest.fixture
def B():
    return torch.tensor([3,87,3])

def test_ncon_to_labels(A, B):
    lhs, rhs, intratr = labels.ncon_to_labels([A, B])

    assert lhs == [A, B]
    assert torch.allclose(rhs, torch.tensor([1, 8, 87]))

    assert len(intratr) == 2
    assert torch.allclose(intratr[0], torch.tensor([-5]))
    assert intratr[1].shape == (0,)  #  is torch.tensor([], dtype=torch.int64)
