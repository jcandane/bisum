"""
dense_intra.py

This contains functions for processing dense tensor intra-operations (slicing & tracing).
"""

import torch
from .generic_functions import first_occurrence_mask, pytorch_delete, iargsort

@torch.jit.script
def remove_ones(a):
    new_shape = torch.tensor( a.shape, dtype=torch.int64 , device=a.device)
    new_shape = new_shape[new_shape!=1]
    return a.reshape([int(i.item()) for i in new_shape])

### given LHS && intra-traces compute reduced/sliced sparse-tensor
@torch.jit.script
def den_post_trans(a, rhs, RHS):
    if torch.numel(RHS)>0:
        rhs=rhs[first_occurrence_mask(rhs)]
        j  = torch.argsort(rhs)
        k  = torch.argsort(RHS)
        ik = iargsort(k)
        return torch.permute(a, [int(i.item()) for i in j[ik]])
    else: ## a is a scalar (cannot permute)
        return a

@torch.jit.script
def den_post_intraTr(a, label):
    """
    GIVEN : a (torch.tensor{dense})
            label (1d-int-torch.tensor)
            intratrace (1d-int-torch.tensor)
    GET   : a (torch.tensor{dense}, intratraced and sliced)
            label (1d-int-torch.tensor, with duplicates and intraTr removed, keeping org. order)
    """
    a = remove_ones(a) ### remove axes which have dimension 1
    b = torch.unique(label)
    if (b.shape == label.shape): ## no repeats proceed
        return a
    #if torch.numel(label)>0: ## if not empty, its a tensor....
    else:
        in_string = ""
        for i in label:
            in_string+=chr(i)

        out_string = ""
        label = label[first_occurrence_mask(label)]
        for i in label:
            out_string+=chr(i)
        return torch.einsum(in_string + "->" + out_string, a) #, label
    #else: ## its a scalar return itself
    #    return a

@torch.jit.script
def den_tensor_intraTr(a, label, intratrace):
    """
    GIVEN : a (torch.tensor{dense})
            label (1d-int-torch.tensor)
            intratrace (1d-int-torch.tensor)
    GET   : a (torch.tensor{dense}, intratraced and sliced)
            label (1d-int-torch.tensor, with duplicates and intraTr removed, keeping org. order)
    """

    in_string = ""
    for i in label:
        in_string+=chr(i)

    out_string = ""
    label = label[first_occurrence_mask(label)]
    for i in label:
        if torch.any(i==intratrace): ## element-wise search
            label = pytorch_delete(label, i==label)
        else:
            out_string+=chr(i)

    return torch.einsum(in_string + "->" + out_string, a)

### TEST
#A    = torch.rand((3,3,3,3,3,3,3,3))
#iqwe = torch.tensor([67, 84, 101, 101, 84, 85, 84, 84])
#print( den_tensor_intraTr(A, iqwe, torch.tensor([])) )