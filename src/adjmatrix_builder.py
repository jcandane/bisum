import torch

###################################
########### adj-matrix ############
###################################

@torch.jit.script
def adjmatrix_builder(labelA, labelB):
    """
    Now that we have unique label characters (no intra-slices nor intra-traces)
    GIVEN : labelA & labelB (each 1d-int-torch.tensor)
    GET   : 2d-int-torch.tensor (of shape (2,n)), where n is the number of traces
    """
    return torch.stack( torch.where((labelA.unsqueeze(1) == labelB.unsqueeze(0))))

## TESTS
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)

#a = torch.unique(torch.randint(0, 8, [int(5)], device=device))
#b = torch.unique(torch.randint(0, 8, [int(5)], device=device))
#print(a,b)
#print(torch.stack( torch.where((a.unsqueeze(1) == b.unsqueeze(0))) , axis=1))
#print(adjmatrix_builder(a, b)) 