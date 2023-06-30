import torch

from uniform_random_sparse_tensor import uniform_random_sparse_tensor

###################################
############# SLICER ##############
###################################

@torch.jit.script
def iargsort(i):
    ## https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python @jesse
    i_rev    = torch.zeros(i.shape, dtype=i.dtype)
    i_rev[i] = torch.arange(len(i))
    return i_rev

@torch.jit.script
def first_occurrence_mask(tensor):
    j = torch.argsort(tensor)
    i = iargsort(j)
    return (torch.concat((torch.ones(1, dtype=tensor.dtype),torch.diff( tensor[j] ))) != 0)[i]

@torch.jit.script
def rowslice(a, label, intratrace=torch.zeros(0,dtype=torch.int32)): ## intratrace=[-5] ## characters...
    """ dimensionally reduce sparse-tensor via a label
    (we need to intratrace)
    GIVEN :   a (torch.tensor !!!Sparse!!!)
              label (1d torch.tensor)
    GET   :   a (sparse-tensor)
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