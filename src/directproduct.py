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
    """
    if ((not a.is_sparse) and (not b.is_sparse)):
        return torch.tensordot(a, b, dims=(0))
    else:
        if (not a.is_sparse):
            a=a.to_sparse()
        if (not b.is_sparse):
            b=b.to_sparse()
        return sparse_outer(a,b)