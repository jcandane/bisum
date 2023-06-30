import torch

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
    denseshape = torch.concat( (torch.flip( torch.cumprod( torch.roll(torch.flip(denseshape, [0]), 1)[1:] , dim=0 ), [0] ), torch.ones(1, dtype=torch.int)),0)
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

    denseshape = torch.concat( (torch.flip( torch.cumprod( torch.roll(torch.flip(denseshape, [0]), 1)[1:] , dim=0 ), [0] ), torch.ones(1, dtype=torch.int)),0)
    denseshape = denseshape.type(list_of_tuples.dtype)
    return torch.matmul(denseshape, list_of_tuples ) ## vector @ matrix product, (n) @ (n, N) , with O ~ nN (linear in N)

#### TESTS
#A       = torch.reshape(torch.randint(0, 15, (18,)), (1,1,18))
#shaper  = torch.tensor([2,4,6])
#print( torch.allclose( torch.flatten(A), tuples_to_ints(ints_to_tuples(A, shaper), shaper)) )