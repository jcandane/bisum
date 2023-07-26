"""
labels.py

Einsum-string and ncon Label processing.
"""

import torch
from generic_functions import first_occurrence_mask
from typing import List

@torch.jit.script
def einsumstr_to_labels(einsum_string : str, device : torch.device=torch.device("cpu")):
    """
    GIVEN : einsum_string (str, numpy.einsum-string)
    GET   : LHS (List[1d-int-torch.tensors] labelling each tensor in einsum_string)
            RHS (1d-int-torch.tensor labelling external indices of output tensor)
            intratraces (List[1d-int-torch.tensors] labelling each tensor, in intra-interal-indices)
    """

    liststring    = einsum_string.replace(" ","").split("->") ## this is a list (at most 2 entries: LHS, RHS)
    if len(liststring)==2: ## RHS given
        LHS = liststring[0].split(",")
        LHS = [torch.tensor([ord(char) for char in word], device=device) for word in LHS]
        RHS =  torch.tensor([ord(char) for char in liststring[1]], device=device)

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
        LHS = [torch.tensor([ord(char) for char in word], device=device) for word in LHS]

        ### build RHS-label
        RHS   = liststring[0].replace(",","")
        RHS   = torch.tensor([ord(char) for char in RHS], device=device)
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

    #remove duplicates
    labelA = LHS[0][first_occurrence_mask(LHS[0])]
    #remove intratraces
    labelA = labelA[torch.logical_not(torch.any((labelA.unsqueeze(1) == intratraces[0].unsqueeze(0)), dim=1))]

    #remove duplicates
    labelB = LHS[1][first_occurrence_mask(LHS[1])]
    #remove intratraces
    labelB = labelB[torch.logical_not(torch.any((labelB.unsqueeze(1) == intratraces[1].unsqueeze(0)), dim=1))]
    #remove inter-traces....

    As_frees = torch.any((labelA.unsqueeze(1) == RHS.unsqueeze(0)), dim=1) ## frees
    Bs_frees = torch.any((labelB.unsqueeze(1) == RHS.unsqueeze(0)), dim=1)
    rhs      = torch.concat( (labelA[As_frees], labelB[Bs_frees]) )
    
    onlydumb = torch.logical_and(torch.logical_not(As_frees).unsqueeze(1), torch.logical_not(Bs_frees).unsqueeze(0))
    interdum = (labelA.unsqueeze(1) == labelB.unsqueeze(0)) ## everything 
    adjmatrix= torch.stack( torch.where(( torch.logical_and(onlydumb, interdum)  )) )

    lhs = [labelA, labelB]
    return LHS, RHS, lhs, rhs, intratraces, adjmatrix

##
## NCON LIST
##

@torch.jit.script
def ncon_to_labels(ncon : List[torch.Tensor]):
    """
    GIVEN : einsum_string (np.einsum string)
    GET   : LHS (List[1d-int-torch.tensors] labelling each tensor in einsum_string)
            RHS (1d-int-torch.tensor labelling external indices of output tensor)
            intratraces (List[1d-int-torch.tensors] labelling each tensor, in intra-interal-indices)
    """

    ### build LHS
    LHS = ncon

    ### build RHS-label
    RHS   = torch.concat((LHS))
    unique_values, counts = torch.unique(RHS, return_counts=True)
    dupes = unique_values[torch.logical_or((counts > 1),(unique_values<0))]  # filter-out duplicate values
    mask  = torch.logical_not(torch.any(torch.eq(RHS.unsqueeze(1), dupes.unsqueeze(0)), 1))
    RHS   = RHS[mask] ## in org. order

    global_internal, counts = torch.unique(torch.concat([torch.unique(lhs) for lhs in LHS]), return_counts=True)
    global_internal = global_internal[counts == len(LHS)]
    not_these       = torch.concat([global_internal, RHS]) ## for intraintersect

    intratraces=[]
    for lhs in LHS:
        unique_values, counts = torch.unique(lhs, return_counts=True) ## intra-dupes
        dupes = unique_values[torch.logical_or((counts > 1), (unique_values<0))]
        dupes = dupes[torch.all(torch.ne(dupes.unsqueeze(1), not_these.unsqueeze(0)),1)]
        intratraces.append( torch.unique(dupes) )

    return LHS, RHS, intratraces

## TEST 
#ncon_example = [torch.tensor([1,8,-5,3]), torch.tensor([3,87,3])]

#lhs, rhs, intratr = ncon_to_labels(ncon_example)
#print(lhs)
#print(rhs)
#print(intratr)