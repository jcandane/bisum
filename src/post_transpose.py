import torch
from adjmatrix_builder import adjmatrix_builder
from _basicfunctions import pytorch_delete, iargsort, lexsort

## test methods
from uniform_random_sparse_tensor import uniform_random_sparse_tensor

@torch.jit.script
def post_transpose(c, labelA, labelB, adj_matrix, RHS : torch.Tensor = torch.zeros(0, dtype=torch.int32)):
    """
    GIVEN : c (torch.tensor)
            labelA & labelB (labels of org. 2 tensors, 1d-int-torch.tensor)
            adj_matrix (between org. 2 tensors, 2d-int-torch.tensor)
            *RHS (if given, 1d-int-torch.tensor)
    GET   : c (torch.tensor transposed into desired order)
    """
    if torch.numel(RHS)==0:
        return c
    else:
        rhs_smash = torch.concat([ pytorch_delete(labelA, adj_matrix[0]) , pytorch_delete(labelB, adj_matrix[1]) ])
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

## TEST
#M = uniform_random_sparse_tensor(15, torch.tensor([12,4,8]))
#a = torch.unique(torch.randint(0, 8, [int(5)]))
#b = torch.unique(torch.randint(0, 8, [int(5)]))
#adjmatrix, rhss = adjmatrix_builder(a, b)
#print( post_transpose(M, a, b, adjmatrix, RHS=None) )
