"""
sparse_intra.py

This contains functions for processing sparse tensor intra-operations (slicing & tracing).
"""

import torch
from generic_functions import first_occurrence_mask, lexsort, iargsort

### given LHS && intra-traces compute reduced/sliced sparse-tensor

@torch.jit.script
def spa_post_trans(a, rhs, RHS):
    rhs=rhs[first_occurrence_mask(rhs)]
    j = torch.argsort(rhs)
    k = torch.argsort(RHS)
    ik= iargsort(k)
    m = j[ik]

    cc= a._indices()[m,:]
    s = torch.tensor(a.shape)[m]
    n = lexsort( cc )
    return torch.sparse_coo_tensor( cc[:,n] , a._values()[n], [int(i.item()) for i in s])

@torch.jit.script
def spa_post_intraTr(a, label):

    uniques, counts = torch.unique(label, return_counts=True)
    if torch.all(counts==1): ## no duplicates in label
        return a #, label
    else: ## slice and wellorder after then
        index = a._indices()
        data  = a._values()
        shape_= torch.tensor(a.shape)

        ## row-wise slice
        for uni in uniques[counts>1]:
            lookat = torch.where(uni==label)[0]

            expand = index[lookat[0]].unsqueeze(0).expand_as(index[lookat[1:]])
            result = (index[lookat[1:]] == expand)
            I      = torch.all(result, dim=0)
            index  = index[:,I] ## row-wise-chop
            data   = data[I]    ## row-wise-chop
        
        ## column-wise slice, intra-external indices
        keep  = first_occurrence_mask(label)
        index = index[keep] ## chop list-of-tuples
        label = label[keep] ## chop label
        shape_= shape_[keep]
        size  = [int(elem) for elem in shape_]
        
        if torch.numel(index)!=0: ## index is empty
            #print(index.shape)
            i     = lexsort(index)
            index = index[:,i]
            data  = data[i]

            domains = torch.concat([torch.ones_like(label[0].reshape((1)), dtype=torch.int32), torch.any((torch.diff( index , dim=1) != 0), dim=0).long()])

            ## do indices
            index = index[:,domains!=0]

            ## do data
            domains = (torch.cumsum(domains, 0)-1)
            ZZ      = torch.zeros_like(index[0], dtype=a.dtype)
            data    = ZZ.scatter_add(0, domains, data)

            return torch.sparse_coo_tensor(index, data, size) #, label
        else:
            data = torch.unsqueeze(torch.unsqueeze(data, 0),0)
            return torch.sparse_coo_tensor(torch.zeros_like(data[:,:,0], dtype=torch.int64), torch.sum(data), [int(b) for b in torch.ones_like(data[:,0,0])])


@torch.jit.script
def spa_tensor_intraTr(a, label, intratrace):

    uniques, counts = torch.unique(label, return_counts=True)
    if torch.all(counts==1) and torch.numel(intratrace)==0: ## no duplicates in label
        return a #, label
    else: ## slice and wellorder after then
        index = a._indices()
        data  = a._values()
        shape_= torch.tensor(a.shape)

        ## row-wise slice
        for uni in uniques[counts>1]:
            lookat = torch.where(uni==label)[0]

            expand = index[lookat[0]].unsqueeze(0).expand_as(index[lookat[1:]])
            result = (index[lookat[1:]] == expand)
            I      = torch.all(result, dim=0)
            index  = index[:,I] ## row-wise-chop
            data   = data[I]    ## row-wise-chop
        
        ## column-wise slice, intra-external indices
        keep  = first_occurrence_mask(label)
        index = index[keep] ## chop list-of-tuples
        label = label[keep] ## chop label
        shape_= shape_[keep]

        ## column-wise slice, remove intratrace
        if torch.numel(intratrace)>0:
            for j in intratrace: ## each column
                keep   = (label!=j)
                label  = label[keep]
                index  = index[keep]
                shape_ = shape_[keep]
            size = [int(elem) for elem in shape_]
        else:
            size = [int(elem) for elem in shape_]
    
        if index.shape[0]==0: ## no more indices just do full sum of data....
            data = torch.unsqueeze(torch.unsqueeze(data, 0),0)
            return torch.sparse_coo_tensor(torch.zeros_like(data[:,:,0], dtype=torch.int64), torch.sum(data), [int(b) for b in torch.ones_like(data[:,0,0])])
        else: ## do partial-trace/sum
            ### lex-order index array
            i     = lexsort(index)
            index = index[:,i]
            data  = data[i]

            domains = torch.concat([torch.ones_like(label[0].reshape((1)), dtype=torch.int32), torch.any((torch.diff( index , dim=1) != 0), dim=0).long()])

            ## do indices
            index = index[:,domains!=0]

            ## do data
            domains = (torch.cumsum(domains, 0)-1)
            ZZ      = torch.zeros_like(index[0], dtype=a.dtype)
            data    = ZZ.scatter_add(0, domains, data)

            return torch.sparse_coo_tensor(index, data, size) #, label
    
##@torch.jit.script
def X_spa_tensor_intraTr(index, data, shape_, label, intratrace):
    """ backend-version
    GIVEN   :  ()
    GET     :
    """

    uniques, counts = torch.unique(label, return_counts=True)
    if torch.all(counts==1): ## no duplicates in label
        return index, data, shape_
    else: ## slice and wellorder after then

        ## row-wise slice
        for uni in uniques[counts>1]:
            lookat = torch.where(uni==label)[0]

            expand = index[lookat[0]].unsqueeze(0).expand_as(index[lookat[1:]])
            result = (index[lookat[1:]] == expand)
            I      = torch.all(result, dim=0)
            index  = index[:,I] ## row-wise-chop
            data   = data[I]    ## row-wise-chop
        
        ## column-wise slice, intra-external indices
        keep  = first_occurrence_mask(label)
        index = index[keep] ## chop list-of-tuples
        label = label[keep] ## chop label
        shape_= shape_[keep]

        ## column-wise slice, remove intratrace
        if len(intratrace)!=0: 
            for j in intratrace: ## each column
                keep   = (label!=j)
                label  = label[keep]
                index  = index[keep]
                shape_ = shape_[keep]
            size = [int(elem) for elem in shape_]
        else:
            size = [int(elem) for elem in shape_]
    
        ### lex-order index array
        i     = lexsort(index)
        index = index[:,i]
        data  = data[i]

        domains = torch.concat([torch.ones_like(label[0].reshape((1)), dtype=torch.int32), torch.any((torch.diff( index , dim=1) != 0), dim=0).long()])

        ## do indices
        index = index[:,domains!=0]

        ## do data
        domains = (torch.cumsum(domains, 0)-1)
        ZZ      = torch.zeros_like(index[0], dtype=data.dtype)
        data    = ZZ.scatter_add(0, domains, data)

        return index, data, shape_
    

### TESTS
#from uniform_random_sparse_tensor import uniform_random_sparse_tensor

#shape = torch.tensor([2,2,2,3])
#A = uniform_random_sparse_tensor(torch.prod(shape), shape)
#A = torch.sparse_coo_tensor( torch.concat((A._indices(),A._indices()),axis=1) , torch.concat((A._values(),A._values())), A.shape)
#print(sp_tensor_intraTr(A, torch.tensor([2, 2, 3, 7]), intratrace=torch.tensor([2])) )