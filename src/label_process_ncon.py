import torch
from typing import List

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