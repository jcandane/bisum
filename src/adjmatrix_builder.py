import torch

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
########### adj-matrix ############
###################################

@torch.jit.script
def adjmatrix_builder(labelA, labelB):
    """
    Now that we have unique label characters (no intra-slices nor intra-traces)
    GIVEN : labelA & labelB ()
    GET   : 2d int torch.tensor (of shape (2,n)), where n is the number of traces
    """
    adjmatrix = torch.stack( torch.where((labelA.unsqueeze(1) == labelB.unsqueeze(0))))
    #rhs_smash = torch.concat([ pytorch_delete(labelA, adjmatrix[0]) , pytorch_delete(labelB, adjmatrix[1]) ])
    return adjmatrix #, rhs_smash

## TESTS
#a = torch.unique(torch.randint(0, 8, [int(5)]))
#b = torch.unique(torch.randint(0, 8, [int(5)]))
#print(a,b)
#print(torch.stack( torch.where((a.unsqueeze(1) == b.unsqueeze(0))) , axis=1))
#print(adjmatrix_builder(a, b))

