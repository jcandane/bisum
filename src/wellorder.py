import torch
from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from _basicfunctions import lexsort

###################################
############ WELLORDER ############
###################################

@torch.jit.script
def wellorder(a):
    """ ONLY applies to sparse tensors
    GIVEN : a (torch.tensor{sparse}, unordered & with duplicates)
    GET   : a (torch.tensor{sparse}, lexordered & without duplicates)
    """

    if a.is_coalesced():
        return a

    i  = lexsort(a._indices())
    II = a._indices()[:,i]

    domains = torch.concat([torch.ones(1, dtype=torch.int32), torch.any((torch.diff( II , dim=1) != 0), dim=0).long()])

    ## do indices
    II = II[:,domains!=0]

    ## do data
    domains = torch.cumsum(domains, 0)-1
    ZZ = torch.zeros_like(II[0], dtype=a.dtype)
    ZZ = ZZ.scatter_add(0, domains, a._values()[i])

    return torch.sparse_coo_tensor(II, ZZ, a.shape)

## TESTS
#shape = torch.tensor([2,2,2,3])
#A = uniform_random_sparse_tensor(torch.prod(shape), shape)
#A = torch.sparse_coo_tensor( torch.concat((A._indices(),A._indices()),axis=1) , torch.concat((A._values(),A._values())), A.shape)
#print(wellorder(A), wellorder(A)._values().shape[0]==torch.prod(shape))