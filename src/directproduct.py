"""
directproduct.py

It computes the direct produc of 2 sparse or dense tensors
output is a sparse XOR dense tensor depending on type of product.
"""

import torch
from generic_functions import cartesian_product

@torch.jit.script
def sparse_outer(a,b):
    """
    GIVEN : a,b (torch.sparse_coo_tensor)
    GET   : c   (torch.sparse_coo_tensor)
    """
    data = torch.outer(a._values(), b._values()).reshape(-1)
    inde = cartesian_product(a._indices(),b._indices())
    size = torch.concat((torch.tensor(a.shape),torch.tensor(b.shape)))
    size = [int(elem.item()) for elem in size]
    return torch.sparse_coo_tensor(inde, data, size) #return [inde, data, size]

@torch.jit.script
def directproduct(a,b):
    """
    tensor product of 2 torch.tensors, resulting tensor is ordered according to the order of tensors
    GIVEN : a,b (torch.tensors{sparse or dense})
    GET   : torch.tensor{sparse or dense
    """
    if ((not a.is_sparse) and (not b.is_sparse)):
        return torch.tensordot(a, b, dims=(0))
    else:
        if (not a.is_sparse):
            a=a.to_sparse()
        if (not b.is_sparse):
            b=b.to_sparse()
        return sparse_outer(a,b)