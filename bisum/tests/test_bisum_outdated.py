"""
TESTS.py

This python-file contains various tests for the bisum package. These tests are outdated and are
superceded by tests in test_bisum.py.
"""

import torch
import pytest

from bisum.bisum import bisum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# set up tensors for all tests (do NOT modify the tensors), treat them
# as readonly; if tensors should be changed then change scope to
# "function"
@pytest.fixture(scope="module")
def A_big():
    return torch.rand(10,10,10,10,10,10, device=device)

@pytest.fixture(scope="module")
def A():
    return torch.rand(10,10,10, device=device)

@pytest.fixture(scope="module")
def B():
    return torch.rand(10,10,10, device=device)

@pytest.fixture(scope="module")
def C():
    return torch.rand(13,13,13, device=device)



@pytest.mark.parametrize("einsumstr", [
    "aeacec,cdd -> aed",  #SPARSE-SPARSE, NO post-inter-externals, NO post-transpose
])
def test_sparse_sparse_no_post_inter_externals(A_big, B, einsumstr):
    assert torch.allclose(
        bisum(einsumstr, A_big.to_sparse(), B.to_sparse()).to_dense(),
        torch.einsum(einsumstr, A_big, B))

@pytest.mark.parametrize("einsumstr", [
    "daa,aed -> a", # SPARSE-SPARSE, YES post-inter-externals, NO post-transpose
    "daa,aed -> ea", ## SPARSE-SPARSE, YES post-inter-externals, YES post-transpose
])
def test_sparse_sparse_with_post_inter_externals(A, B, einsumstr):
    ### acc -> ac
    ### cdd -> cd  ----> accd ----> acd

    ### acc -> ac
    ### cdd -> cd  ----> accd ----> acd

    assert torch.allclose(
        bisum(einsumstr, A.to_sparse(), B.to_sparse()).to_dense(),
        torch.einsum(einsumstr, A, B))


def test_sparse_sparse_with_empty_array_1(C, B):
    # if slicing returns an empty array avoid lexsort!!!!!!!!!!!!!!!! AND "dad,mom -> dm"
    einsumstr = "aaa,wop -> a" ## SPARSE-SPARSE, YES post-inter-externals, YES post-transpose
    assert torch.allclose(
        bisum(einsumstr, C.to_sparse(), B.to_sparse()).to_dense(),
        torch.einsum(einsumstr, C, B))

@pytest.mark.parametrize("einsumstr", [
    "aaa,wop -> a", ## SPARSE-DENSE, YES post-inter-externals, YES post-transpose
    "qaq,wow -> ",  ## SPARSE-DENSE, YES post-inter-externals, YES post-transpose
    ])
def test_sparse_dense_with_empty_array_2(A, B, einsumstr):
    # if slicing returns an empty array avoid lexsort!!!!!!!!!!!!!!!! AND "dad,mom -> dm"
    assert torch.allclose(
        bisum(einsumstr, A.to_sparse(), B).to_dense(),
        torch.einsum(einsumstr, A, B))
