import torch

## test methods
from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from adjmatrix_builder import adjmatrix_builder

###################################
######### PyTorch DELETE ##########
###################################

@torch.jit.script
def pytorch_delete(data, args_to_delete):
    """
    *** delete entries of a torch.tensor given indices (to_delete)
    GIVEN : torch.tensor
    GET   : torch.tensor (without deleted entries)
    """
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[args_to_delete] = False
    return data[mask]

@torch.jit.script
def iargsort(i):
    i_rev    = torch.zeros(i.shape, dtype=i.dtype)
    i_rev[i] = torch.arange(len(i))
    return i_rev

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

def post_transpose(c, labelA, labelB, adj_matrix, RHS : None = None):
    """
    GIVEN : c (torch.tensor)
            labelA & labelB (labels of org. 2 tensors)
            adj_matrix (between org. 2 tensors)
            *RHS (if given)
    GET   : 
    """
    if RHS is None:
        return c
    else:
        rhs_smash = torch.concat([ pytorch_delete(labelA, adjmatrix[0]) , pytorch_delete(labelB, adjmatrix[1]) ])
        if torch.all(RHS==rhs_smash):
            return c
        else:
            j = torch.argsort(rhs_smash)
            k = torch.argsort(RHS)
            ik= iargsort(k)
            m = permute = j[ik]
            if c.is_sparse:
                cc= c._indices()[m,:]
                s = torch.tensor(c.shape)[m]
                n = lexsort( cc )
                return torch.sparse_coo_tensor( cc[:,n] , c._values()[n], s)
            else: # dense
                return torch.permute(c, permute)

M = uniform_random_sparse_tensor(15, torch.tensor([12,4,8]))
a = torch.unique(torch.randint(0, 8, [int(5)]))
b = torch.unique(torch.randint(0, 8, [int(5)]))
adjmatrix, rhss = adjmatrix_builder(a, b)

post_transpose(M, a, b, adjmatrix, RHS=None)