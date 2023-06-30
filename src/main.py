import torch
from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from directintersection import twotrace
from label_process_einsum import einsumstr_to_labels
from label_process_ncon import ncon_to_labels
from slice import rowslice
from wellorder import wellorder
from adjmatrix_builder import adjmatrix_builder
from post_transpose import post_transpose

def sTr(Rx, a, b):

    if isinstance(Rx, str): # einsum
        LHS, RHS, intratraces = einsumstr_to_labels(Rx)
        print(intratraces[0], intratraces[1])
        a, la = rowslice(a, LHS[0], intratrace=intratraces[0])
        b, lb = rowslice(b, LHS[1], intratrace=intratraces[1])
        
        a     = wellorder(a)
        b     = wellorder(b)
        A     = adjmatrix_builder(la, lb)
        c = twotrace(a, b, A)
        return post_transpose(c, la, lb, A)

    else:
        if isinstance(Rx, list): # ncon  (no post transpose)
            LHS, RHS, intratraces = ncon_to_labels(Rx)
            a, la = rowslice(a, LHS[0], intratrace=intratraces[0])
            b, lb = rowslice(b, LHS[1], intratrace=intratraces[1])
            a     = wellorder(a)
            b     = wellorder(b)
            A     = adjmatrix_builder(la, lb)
            return twotrace(a, b, A)
        
        else:
            if torch.is_tensor(Rx): # adjmatrix (no post transpose)
                return twotrace(a, b, Rx)
            else:
                raise ValueError("tracing Rx is not valid")

A = uniform_random_sparse_tensor(150, torch.tensor([12,13,14]))
B = uniform_random_sparse_tensor(165, torch.tensor([13,13,14]))

sTr(torch.tensor([[1,0]]).T, A, B)
sTr("ijk,ljm", A, B)
sTr("ijk,ljm->km", A, B)
sTr([torch.tensor([0,1,-3]),torch.tensor([1,1,7])], A, B)

print(sTr([torch.tensor([0,1,-3]),torch.tensor([1,1,7])], A, B).shape)
print(sTr(torch.tensor([[1,0]]).T, A, B).shape)
print(sTr("ijk,ljm->km", A, B).shape)