"""
___directproduct.py

Internal Version of directproduct.py . It computes the direct produc of 2 sparse or dense tensors
i.e. sparse-tensors are given as List[torch.tensors & List[int]] .
"""

import torch
from generic_functions import cartesian_product

@torch.jit.script
def sparse_outer(a_index, a_data, a_shape, b_index, b_data, b_shape):
    """
    GIVEN : a,b (torch.sparse_coo_tensor)
    GET   : c   (torch.sparse_coo_tensor)
    """
    data = torch.outer(a_data, b_data).reshape(-1)
    inde = cartesian_product(a_index, b_index)
    size = a_shape + b_shape #torch.concat((torch.tensor(a.shape),torch.tensor(b.shape)))
    return [inde, data, size] 

@torch.jit.script
def directproduct(a_index, a_data, a_shape, b_index, b_data, b_shape):
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