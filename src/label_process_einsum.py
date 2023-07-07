import torch

@torch.jit.script
def einsumstr_to_labels(einsum_string : str):
    """
    GIVEN : einsum_string (str, numpy.einsum-string)
    GET   : LHS (List[1d-int-torch.tensors] labelling each tensor in einsum_string)
            RHS (1d-int-torch.tensor labelling external indices of output tensor)
            intratraces (List[1d-int-torch.tensors] labelling each tensor, in intra-interal-indices)
    """

    liststring    = einsum_string.replace(" ","").split("->") ## this is a list (at most 2 entries: LHS, RHS)
    if len(liststring)==2: ## RHS given
        LHS = liststring[0].split(",")
        LHS = [torch.tensor([ord(char) for char in word]) for word in LHS]
        RHS =  torch.tensor([ord(char) for char in liststring[1]])

        global_internal, counts = torch.unique(torch.concat([torch.unique(lhs) for lhs in LHS]), return_counts=True)
        global_internal = global_internal[counts == len(LHS)]
        not_these       = torch.concat([global_internal, RHS]) ## for intraintersect

        intratraces=[] ## over each value of
        for lhs in LHS:
            unique_values, counts = torch.unique(lhs, return_counts=True) ## intra-dupes
            unique_values = unique_values[torch.all(torch.ne(unique_values.unsqueeze(1), not_these.unsqueeze(0)),1)]
            intratraces.append( torch.unique(unique_values) )

    else: #if len(liststring)==1: ## no RHS, go reg. convention (repeats are dummies)
        LHS = liststring[0].split(",") ## should be at most 2-here
        LHS = [torch.tensor([ord(char) for char in word]) for word in LHS]

        ### build RHS-label
        RHS   = liststring[0].replace(",","")
        RHS   = torch.tensor([ord(char) for char in RHS])
        unique_values, counts = torch.unique(RHS, return_counts=True)
        dupes = unique_values[counts > 1]     # Filter-out duplicate values, gather
        mask  = torch.logical_not(torch.any(torch.eq(RHS.unsqueeze(1), dupes.unsqueeze(0)), 1))
        RHS   = RHS[mask] ## in org. order

        global_internal, counts = torch.unique(torch.concat([torch.unique(lhs) for lhs in LHS]), return_counts=True)
        global_internal = global_internal[counts == len(LHS)]
        not_these       = torch.concat([global_internal, RHS]) ## for intraintersect

        # Perform element-wise XOR comparison
        intratraces=[] ## over each value of
        for lhs in LHS:
            unique_values, counts = torch.unique(lhs, return_counts=True) ## intra-dupes
            dupes = unique_values[counts > 1]
            dupes = dupes[torch.all(torch.ne(dupes.unsqueeze(1), not_these.unsqueeze(0)),1)]
            intratraces.append( torch.unique(dupes) )

    return LHS, RHS, intratraces

##TESTS
#from _randomeinsumstring import random_einsum_string
#
#l_rand = random_einsum_string(rhs=True)

#lhs, rhs, intratr = (einsumstr_to_labels(l_rand))
#print(lhs)
#print(rhs)
#print(intratr)