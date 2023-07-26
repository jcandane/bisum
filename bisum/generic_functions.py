"""
generic_functions.py

This python file contains defs/functions that could be widely used in many codes.
"""

import torch

###################################
######## INVERSE ARGSORT  #########
###################################

@torch.jit.script
def iargsort(i):
    """
    ***suppose we have a unsorted-array A, its args to sort are obtained via:
    i = argsort(A)
    *its this indices 'i' that are used below, to obtain the inverse, i.e.
    A_sorted = A[i]
    A_back   = A_sorted[i_rev] = A[i][i_rev]
    ***
    GIVEN : i (argsort-indices)
    GET   : i_rev ()
    """
    i_rev    = torch.ones_like(i, dtype=i.dtype)
    i_rev[i] = torch.arange(len(i), dtype=i.dtype, device=i.device)
    return i_rev

###################################
######### 1st Occurrence ##########
###################################

@torch.jit.script
def first_occurrence_mask(tensor):
    j = torch.argsort(tensor)
    i = iargsort(j)
    return (torch.concat((torch.ones(1, dtype=tensor.dtype, device=tensor.device),torch.diff( tensor[j] ))) != 0)[i]

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

###################################
############# LEXSORT #############
###################################

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

###################################
######### SPARSE RESHAPER #########
###################################

@torch.jit.script
def ints_to_tuples(the_ints, denseshape):
    """
    ***Reshape single sparse list-of-tuples 1d tensor/array into a 2d array/tensor list-of-tuples.
    GIVEN : the_ints (int torch.tensor)
            denseshape (int 1d torch.tensor)
    GET   : 2d torch.tensor (being the 'list'-of-tuples)
    """
    the_ints   = torch.flatten(the_ints) ## forces the_ints to be 1-dimensional
    denseshape = torch.flatten(denseshape)
    denseshape = torch.concat( (torch.flip( torch.cumprod( torch.roll(torch.flip(denseshape, [0]), 1)[1:] , dim=0 ), [0] ), torch.ones_like(denseshape.reshape((denseshape.shape[0],1))[0], dtype=torch.int)),0)
    out_tuple  = []

    for s in denseshape: ## for each column in new-shape, generates tuple....
        out_tuple.append( the_ints // s )
        the_ints = torch.remainder( the_ints, s )
    return torch.stack(out_tuple, 0)

@torch.jit.script
def tuples_to_ints(list_of_tuples, denseshape):
    """
    ***Reshape sparse list-of-tuples tensor/array into a 1d array/tensor.
    GIVEN:  list_of_tuples (2d int torch.tensor, with shape (col'n,rows), eg (3,12357))
            dense_shape (1d torch.tensor, shape of the dense representation)
    GET:    1d int torch.tensor (corresponding to pair-function)
    """
    denseshape = torch.flatten(denseshape)
    if list_of_tuples.shape[0]!=denseshape.shape[0]:
        if torch.numel(list_of_tuples)==0:
            raise ValueError("list_of_tuples is empty")
        else:
            raise ValueError("tuple-shapes and shape must match")

    denseshape = torch.concat( (torch.flip( torch.cumprod( torch.roll(torch.flip(denseshape, [0]), 1)[1:] , dim=0 ), [0] ), torch.ones_like(denseshape.reshape((denseshape.shape[0],1))[0], dtype=denseshape.dtype)),0)
    denseshape = denseshape.type(list_of_tuples.dtype)
    #return torch.matmul(denseshape, list_of_tuples ) ## vector @ matrix product, (n) @ (n, N) , with O ~ nN (linear in N) ### this isnt implemented in cuda for pytorch!!!!! only works for int32
    return torch.sum(denseshape.unsqueeze(1) * list_of_tuples, 0) ## matrix-product
    #return torch.matmul(list_of_tuples.T, denseshape ) 

#### TESTS 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#print(device)

#A       = torch.reshape(torch.randint(0, 15, (18,), device=device), (1,1,18))
#shaper  = torch.tensor([2,4,6], device=device)
#print( torch.allclose( torch.flatten(A), tuples_to_ints(ints_to_tuples(A, shaper), shaper)) )

##### tester!!!
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#B = torch.reshape(torch.randint(0, 15, (18,), device=device, dtype=torch.int32), (1,1,18))
#sh= torch.tensor([2,4,6], device=device, dtype=torch.int32)

#print( torch.allclose( torch.flatten(A), tuples_to_ints(ints_to_tuples(A, shaper), shaper)) )

##########################################
############## OUTERPRODUCT ##############
##########################################

@torch.jit.script
def cartesian_product(a, b):
    ab = (a.unsqueeze(1) * torch.ones(b.shape[1], dtype=a.dtype, device=a.device).unsqueeze(1) ).swapaxes(1,2).reshape((a.shape[0], a.shape[1] * b.shape[1]))
    ba = (b.unsqueeze(1) * torch.ones(a.shape[1], dtype=b.dtype, device=a.device).unsqueeze(1) ).reshape((b.shape[0], b.shape[1] * a.shape[1]))
    return torch.cat((ab, ba), dim=0)
