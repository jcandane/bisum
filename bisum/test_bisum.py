"""
TESTS.py

This python-file contains various tests for the bisum package.
"""

import torch
from bisum import bisum
###from dev_defs import sptensordot, sdtensordot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

#### Einsum Sparse-Sparse
#einsumstr, shape1, shape2 = random_einsum_string(return_shapes=True)


#C = torch.einsum( einsumstr, A, B )
#c = bisum( einsumstr, A.to_sparse(), B.to_sparse() )

#print(C.shape, c.shape)
#print( einsumstr )
#print( torch.allclose(C,c.to_dense()) )

#### REGULAR TRACE
A = torch.rand(10,10,10,10, device=device)
B = torch.rand(10,10,10,10, device=device)
adjj = torch.tensor([[0,3,2,1],[1,2,3,0]], device=device)

####c = sptensordot(A.to_sparse(), B.to_sparse(), dims=adjj)
#s = sdtensordot(A.to_sparse(), B, dims=adjj)
#S = sdtensordot(A, B.to_sparse(), dims=adjj)
####C = torch.tensordot(A, B, dims=adjj)

####print( torch.allclose(C,c.to_dense()) ) #and torch.allclose(C,s.to_dense()) and torch.allclose(C,S.to_dense()) )

### OUTERPRODUCT
A = torch.rand(10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
adjj = torch.tensor([[],[]], device=device)

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

einsumstr = "aeacec,cdd -> aed" ## SPARSE-SPARSE, NO post-inter-externals, NO post-transpose
A = torch.rand(10,10,10,10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B.to_sparse()).to_dense(), torch.einsum( einsumstr , A, B )) )

einsumstr = "daa,aed -> a" ## SPARSE-SPARSE, YES post-inter-externals, NO post-transpose
A = torch.rand(10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B.to_sparse()).to_dense(), torch.einsum( einsumstr , A, B )) )

### acc -> ac
### cdd -> cd  ----> accd ----> acd

einsumstr = "daa,aed -> ea" ## SPARSE-SPARSE, YES post-inter-externals, YES post-transpose
A = torch.rand(10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B.to_sparse()).to_dense(), torch.einsum( einsumstr , A, B )) )

### acc -> ac
### cdd -> cd  ----> accd ----> acd

##### if slicing returns an empty array avoid lexsort!!!!!!!!!!!!!!!! AND "dad,mom -> dm"
einsumstr = "aaa,wop -> a" ## SPARSE-SPARSE, YES post-inter-externals, YES post-transpose
A = torch.rand(13,13,13, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B.to_sparse()).to_dense(), torch.einsum( einsumstr , A, B )) )


##### if slicing returns an empty array avoid lexsort!!!!!!!!!!!!!!!! AND "dad,mom -> dm"
einsumstr = "aaa,wop -> a" ## SPARSE-DENSE, YES post-inter-externals, YES post-transpose
A = torch.rand(10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B).to_dense(), torch.einsum( einsumstr , A, B )) )

##### if slicing returns an empty array avoid lexsort!!!!!!!!!!!!!!!! AND "dad,mom -> dm"
einsumstr = "qaq,wow -> " ## SPARSE-DENSE, YES post-inter-externals, YES post-transpose
A = torch.rand(10,10,10, device=device)
B = torch.rand(10,10,10, device=device)
print( torch.allclose( bisum( einsumstr, A.to_sparse(), B).to_dense(), torch.einsum( einsumstr , A, B )) )