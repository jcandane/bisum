import torch

from uniform_random_sparse_tensor import uniform_random_sparse_tensor

###################################
##### LEXICOGRAPHIC ORDERING ######
###################################

@torch.jit.script
def lex_lessthan(a, b): ## a<b ? for tuples a & b
    c = b - a
    C = torch.nonzero(c)
    if C.numel()==0: ## special-case: if-equal?
        return False
    else:
        if (c[C[0]] > 0): ## is 1st nonzero positive?
            return True
        else:
            return False

@torch.jit.script
def lexsort(LoT):
    """
    lexicographically-order list-of-tuples (tuple-index, list-index), e.g. (6,123854)
    such that the 0-th entry in LoT is the most important, i.e. (a_0,a_1,a_2,...,a_n)
    GIVEN : LoT (2d torch.tensor)
    GET   : idx (1d torch.tensor : indices to lexsort LoT)
    """
    idx = torch.argsort(LoT[-1], stable=True)
    for k in reversed(LoT[:-1]):
        idx = idx.gather(0, torch.argsort(k.gather(0, idx), stable=True))
    return idx

@torch.jit.script
def nplexsort(LoT):
    """ same as numpy.lexsort a.k.a colexsort
    lexicographically-order list-of-tuples (tuple-index, list-index), e.g. (6,123854)
    such that the n-th entry in LoT is the most important, i.e. (a_0,a_1,a_2,...,a_n)
    GIVEN : LoT (2d torch.tensor)
    GET   : idx (1d torch.tensor : indices to lexsort LoT)
    """
    idx = torch.argsort(LoT[0], stable=True)
    for k in LoT[1:]: ## each axis after 0th
        idx = idx.gather(0, torch.argsort(k.gather(0, idx), stable=True))
    return idx


###################################
############ WELLORDER ############
###################################

@torch.jit.script
def wellorder(a):
    if a.is_coalesced():
        return a

    i  = lexsort(a._indices())
    II = a._indices()[:,i]

    domains = torch.concat([torch.ones(1, dtype=torch.int32), torch.any((torch.diff( II , dim=1) != 0), dim=0).long()])

    ## do indices
    II = II[:,domains!=0]

    ## do data
    domains = torch.cumsum(domains, 0)-1
    ZZ = torch.zeros(domains[-1]+1, dtype=a.dtype)
    ZZ = ZZ.scatter_add(0, domains, a._values()[i])

    return torch.sparse_coo_tensor(II, ZZ, a.shape)

## TESTS
#shape = torch.tensor([2,2,2,3])
#A = uniform_random_sparse_tensor(torch.prod(shape), shape)
#A = torch.sparse_coo_tensor( torch.concat((A._indices(),A._indices()),axis=1) , torch.concat((A._values(),A._values())), A.shape)
#print(wellorder(A), wellorder(A)._values().shape[0]==torch.prod(shape))