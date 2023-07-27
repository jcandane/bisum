# bisum

PyTorch Sparse-Tensor Partial-Trace
This program traces 2 sparse-tensor (torch.tensor objects) via 3 Tracing-Prescription:
1. {einsum} string (like numpy, str, labelling each tensor axis)
2. ncon (used in the tensor-network community, list of 1d int torch.tensor, labelling each tensor axis)
3. adjacency-matrix (as in numpy.tensordot, (2,n) 2d int torch.tensor, with n being the number of indices idenified between the two tensors)

## API

Let's begin by initializing the 2 tensors, we can initialize random-sparse-tensors 
```python
from bisum.bisum import bisum
import torch

shape_A = torch.tensor([8,7,7,4,11,6])
shape_B = torch.tensor([9,7,3,7,11,8])
A = torch.rand(shape_A)
B = torch.rand(shape_B)
```

Suppose we would like to compute the following partial-trace/tensor-contraction $C_{njwl} = A_{iksndj} B_{wklsdi}$:
```python
C_einsum = bisum("iksndj, wklsdi -> njwl", A, B)
C_ncon   = bisum([[-1,-2,-3,4,-5,6],[1,-2,3,-3,-5,-1]], A, B)
C_adjmat = bisum(torch.tensor([[0,1,2,4],[5,1,3,4]]), A, B)

print( torch.allclose(C_einsum, C_ncon) and torch.allclose(C_ncon, C_adjmat) )
```
while the pure tensor-product, $\otimes$ is:
```python
C_einsum = bisum("abcdef, ghijkl", A, B)
C_ncon   = bisum([], A, B)
C_adjmat = bisum(torch.tensor([]), A, B)

print( np.allclose(C_einsum, C_ncon) and np.allclose(C_ncon, C_adjmat) )
```

## Install

```bash
pip install bisum
```

