import torch

##########################################
############## OUTERPRODUCT ##############
##########################################

@torch.jit.script
def cartesian_product(a, b):
    ab = (a.unsqueeze(1) * torch.ones(b.shape[1], dtype=a.dtype).unsqueeze(1) ).swapaxes(1,2).reshape((a.shape[0], a.shape[1] * b.shape[1]))
    ba = (b.unsqueeze(1) * torch.ones(a.shape[1], dtype=b.dtype).unsqueeze(1) ).reshape((b.shape[0], b.shape[1] * a.shape[1]))
    return torch.cat((ab, ba), dim=0)

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
    return torch.sparse_coo_tensor(inde, data, size)

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
    
##########################################
############# INTRA METHODS ##############
##########################################

@torch.jit.script
def sparse_outer_(a_index, a_data, a_shape, b_index, b_data, b_shape):
    """
    GIVEN : a_index
            a_data
            a_shape
            b_index
            b_data
            b_shape
    GET   : c_index
            c_data
            c_shape
    """
    data = torch.outer(a_data, b_data).reshape(-1)
    inde = cartesian_product(a_index, b_index)
    size = torch.concat(((a_shape),(b_shape)))
    return inde, data, size

## TEST
#from uniform_random_sparse_tensor import uniform_random_sparse_tensor

#A = uniform_random_sparse_tensor(4, torch.tensor([3,6]))
#B = uniform_random_sparse_tensor(5, torch.tensor([4,3]))

#print( sparse_outer_(A._indices(), A._values(), torch.tensor(A.shape), B._indices(), B._values(), torch.tensor(B.shape)) )