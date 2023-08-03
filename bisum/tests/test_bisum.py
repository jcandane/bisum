"""
TESTS.py

This python-file contains various tests for the bisum package.
"""

import torch

import pytest

from bisum import bisum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### The commented out code had not been testing anything. It is left here for historical reasons.
### It should be turned into proper pytests.

####c = sptensordot(A.to_sparse(), B.to_sparse(), dims=adjj)
#s = sdtensordot(A.to_sparse(), B, dims=adjj)
#S = sdtensordot(A, B.to_sparse(), dims=adjj)
####C = torch.tensordot(A, B, dims=adjj)

####print( torch.allclose(C,c.to_dense()) ) #and torch.allclose(C,s.to_dense()) and torch.allclose(C,S.to_dense()) )

### OUTERPRODUCT
# A = torch.rand(10,10,10, device=device)
# B = torch.rand(10,10,10, device=device)
# adjj = torch.tensor([[],[]], device=device)

####c = sptensordot(A.to_sparse(), B.to_sparse(), dims=adjj)
####s = sdtensordot(A.to_sparse(), B, dims=adjj)
####S = sdtensordot(A, B.to_sparse(), dims=adjj)
####C = torch.tensordot(A, B, dims=0)

####print( torch.allclose(C,c.to_dense()) and torch.allclose(C,s.to_dense()) and torch.allclose(C,S.to_dense()) )

#einsumstr = "mmQmI,DmmQ -> m"
#einsumstr = "mmQmI,ImmQ -> m"
#einsumstr = "abcde,dcag -> bg" ## DONE ---->having trouble with intratrace!!!
#einsumstr = "abede,dcag -> bg" ## post slice problems....
#einsumstr, shape1, shape2 = random_einsum_string(return_shapes=True)
#A = torch.rand([4,5,5,7,5]) ##[4,4,8,4,9])
#B = torch.rand([7,6,4,9])   ##[2,4,4,8])

#print(einsumstr)
#C = torch.einsum( einsumstr, A, B )
#c = bisum( einsumstr, A.to_sparse(), B.to_sparse() )

#print(C.shape, c.shape)
#print( torch.allclose(C, c.to_dense()) )

### EINSUM TESTS

class TestBisum:
    @pytest.fixture(scope="class")
    @staticmethod
    def A():
        return torch.rand(10,10,10, device=device)

    @pytest.fixture(scope="class")
    @staticmethod
    def B():
        return torch.rand(10,10,10, device=device)

    @pytest.fixture(scope="class")
    @staticmethod
    def A2():
        return torch.rand(10,10,10,10,10,10, device=device)

    @pytest.fixture(scope="class")
    @staticmethod
    def A3():
        return torch.rand(13,13,13, device=device)

    @staticmethod
    def _sparse_sparse(einsumstr, A, B):
        return bisum(einsumstr, A.to_sparse(), B.to_sparse()).to_dense()

    @staticmethod
    def _dense_dense(einsumstr, A, B):
        return bisum(einsumstr, A, B).to_dense()

    @staticmethod
    def _sparse_dense(einsumstr, A, B):
        return bisum(einsumstr, A.to_sparse(), B).to_dense()

    @staticmethod
    def _dense_sparse(einsumstr, A, B):
        return bisum(einsumstr, A, B.to_sparse()).to_dense()

    @pytest.fixture(params=[
        pytest.param(('sparse', 'sparse'), id="sparse/sparse"),
        pytest.param(('sparse', 'dense'), id="sparse/dense"),
        pytest.param(('dense', 'sparse'), id="dense/sparse"),
        pytest.param(('sparse', 'sparse'), id="dense/dense"),
    ])
    def bisum_func(self, request):
        """Return the bisum() invocation with the requested dense/sparse input argument handling."""
        # This look up table is not super-elegant but fairly transparent (and avoids if/elif)
        _f = {('sparse', 'sparse'): self._sparse_sparse,
              ('sparse', 'dense'): self._sparse_dense,
              ('dense', 'sparse'): self._dense_sparse,
              ('dense', 'dense'): self._dense_dense,
              }
        return _f[request.param]

    def _assert_bisum(self, A, B, einsumstr, bisum_func):
        C = bisum_func(einsumstr, A, B)
        ref = torch.einsum(einsumstr , A, B)
        assert torch.allclose(C, ref)


    @pytest.mark.parametrize("einsumstr", [
        "daa,aed -> a",      # id="YES post-inter-externals, NO post-transpose",
        "daa,aed -> ea",     # id="YES post-inter-externals, YES post-transpose",
        "aaa,wop -> a",      # id="empty array/no lexsort, YES post-inter-externals, YES post-transpose",
        "qaq,wow -> ",       # id="empty array/no lexsort, YES post-inter-externals, YES post-transpose",
    ])
    def test_bisum(self, A, B, einsumstr, bisum_func):
        self._assert_bisum(A, B, einsumstr, bisum_func)

    def test_bisum_NO_post_inter_externals_NO_post_transpose(self, A2, B, bisum_func,
                                                                           einsumstr="aeacec,cdd -> aed"):
        # Test needs larger A tensor.
        #### "NO post-inter-externals, NO post-transpose"
        self._assert_bisum(A2, B, einsumstr, bisum_func)


    def test_bisum_no_lexsort(self, A3, B, bisum_func, einsumstr="aaa,wop -> a"):
        # Originall, this was a test with a different A tensor.
        ##### if slicing returns an empty array avoid lexsort!!
        ## SPARSE-SPARSE, YES post-inter-externals, YES post-transpose
        self._assert_bisum(A3, B, einsumstr, bisum_func)
