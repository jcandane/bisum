import torch

from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from _basicfunctions import first_occurrence_mask

###################################  
############# SLICER ##############
###################################

@torch.jit.script
def rowslice(a, label, intratrace=torch.zeros(0,dtype=torch.int32)): ## intratrace=[-5] ## characters...
    """ dimensionally reduce sparse-tensor via a label
    (we need to intratrace)
    GIVEN :   a (torch.tensor{sparse})
              label (1d-int-torch.tensor)
    GET   :   a (torch.tensor{sparse}, sliced to remove duplicate rows/columns defined via label)
    """
    index = a._indices()
    data  = a._values()
    shape_= torch.tensor(a.shape)

    uniques, counts = torch.unique(label, return_counts=True)
    for uni in uniques[counts>1]:
        lookat = torch.where(uni==label)[0]

        expand = index[lookat[0]].unsqueeze(0).expand_as(index[lookat[1:]])
        result = (index[lookat[1:]] == expand)
        I      = torch.all(result, dim=0)
        index  = index[:,I] ## row-wise-chop
        data   = data[I]    ## row-wise-chop

    keep  = first_occurrence_mask(label)
    index = index[keep] ## chop list-of-tuples
    label = label[keep] ## chop label
    shape_= shape_[keep]
    if intratrace.shape[0]!=0: ## remove intratrace
        for j in intratrace:
            keep   = (label!=j)
            label  = label[keep]
            index  = index[keep]
            shape_ = shape_[keep]

    size = [int(elem) for elem in shape_]
    return torch.sparse_coo_tensor(index, data, size), label

## TESTS
#A = uniform_random_sparse_tensor(35, torch.tensor([12,8,8,3]))
#l = torch.tensor([-4,8,8,2]) ## label
#print( rowslice(A, l, intratrace=torch.zeros(0,dtype=torch.int32)) )