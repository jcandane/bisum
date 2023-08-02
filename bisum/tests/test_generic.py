import torch

import pytest

from bisum import generic_functions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module",
                params=[None,
                        pytest.param(int, id="int"),
                        pytest.param(torch.int32, id="torch.int32", marks=pytest.mark.xfail),
                        pytest.param(torch.int64, id="torch.int64"),
                        ])
def A_shaper(request):
    # tensor A and shaper with the same dtype
    A = torch.reshape(torch.randint(0, 15, (18,),
                                        device=device,
                                        dtype=request.param), (1,1,18))
    shaper = torch.tensor([2,4,6], device=device, dtype=request.param)
    return A, shaper


def test_tuples_to_ints(A_shaper):
    A, shaper = A_shaper
    assert torch.allclose(
        generic_functions.tuples_to_ints(
            generic_functions.ints_to_tuples(A, shaper), shaper),
        torch.flatten(A))


