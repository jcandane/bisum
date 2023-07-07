import torch
from _basicfunctions import ints_to_tuples
from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from label_process_einsum import einsumstr_to_labels
from label_process_ncon import ncon_to_labels
from rowslice import rowslice
from adjmatrix_builder import adjmatrix_builder
from wellorder import wellorder
from directintersection import twotrace, make_matrix
from post_transpose import post_transpose

def bisum(Rx, a, b, return_label : bool = False, device : torch.device=torch.device("cpu")):
    """
    GIVEN : Rx (str, list, 2d-torch.tensor)
            a, b (torch.tensor{sparse or dense})
            return_label (bool, choice whether to give resulting label)
    GET   : c (torch.tensor)
            **RHS (1d-torch.tensor)
    """

    if isinstance(Rx, str): # einsum
        ## determine if dense or sparse make an array sparse = [False, True, True, ...]
        LHS, RHS, intratraces = einsumstr_to_labels(Rx)
        a, la = rowslice(a, LHS[0], intratrace=intratraces[0])
        b, lb = rowslice(b, LHS[1], intratrace=intratraces[1])
        a     = wellorder(a)
        b     = wellorder(b)
        A     = adjmatrix_builder(la, lb)
        c     = twotrace(a, b, A)
        c     = post_transpose(c, la, lb, A)
        if return_label:
            return c, RHS
        else:
            return c

    else:
        if isinstance(Rx, list): # ncon  (no post transpose)
            LHS, RHS, intratraces = ncon_to_labels(Rx)

            if a.is_sparse and b.is_sparse: ## a & b SPARSE
                A, lA = rowslice(a, LHS[0], intratrace=intratraces[0])
                B, lB = rowslice(b, LHS[1], intratrace=intratraces[1])
                A     = wellorder(A)
                B     = wellorder(B)

                AA    = adjmatrix_builder(lA, lB)
                A, le = make_matrix(A, AA[0], left=True)
                B, lf = make_matrix(B, AA[1], left=False)

                m_c   = A @ B
                II    = torch.concat([ints_to_tuples(m_c._indices()[0], le), ints_to_tuples(m_c._indices()[1], lf)])
                m_c   = torch.sparse_coo_tensor(II, m_c._values(), [int(i) for i in torch.concat([le, lf])])

            else:
                if (not a.is_sparse) and b.is_sparse: ## A==dense & B==Sparse, must place dense matrix to the right!
                    None

                else:
                    if (a.is_sparse) and (not b.is_sparse): ## A==Sparse & B==dense
                        None
                    else: ## A==dense & B==dense
                        None

            
            a, la = rowslice(a, LHS[0], intratrace=intratraces[0])
            b, lb = rowslice(b, LHS[1], intratrace=intratraces[1])
            a     = wellorder(a)
            b     = wellorder(b)
            A     = adjmatrix_builder(la, lb)
            c     = twotrace(a, b, A)
            if return_label:
                return c, RHS
            else:
                return c

        else:
            if torch.is_tensor(Rx): # adjmatrix (no post transpose)
                c = twotrace(a, b, Rx)
                if return_label:
                    return c, torch.arange(1, 1+c.ndim)
                else:
                    return c
            else:
                raise ValueError("tracing Rx is not valid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

A = uniform_random_sparse_tensor(150, torch.tensor([12,13,14]), device=device)
B = uniform_random_sparse_tensor(165, torch.tensor([13,13,14]), device=device)

bisum(torch.tensor([[1,0]]).T, A, B)
bisum("ijk,ljm", A, B)
bisum("ijk,ljm->km", A, B)
bisum([torch.tensor([0,1,-3]),torch.tensor([1,1,7])], A, B)

bisum("ijk,ljm->km", A, B, return_label=True)

print(bisum([torch.tensor([0,1,-3]),torch.tensor([1,1,7])], A, B).shape)
print(bisum(torch.tensor([[1,0]]).T, A, B).shape)
print(bisum("ijk,ljm->km", A, B).shape)
#
print(bisum("ijk,ljm->mk", A, B).shape) ## print(sTr("ijk,ljm->ml", A, B).shape) <--- this failed!! ohh the shape and labels must match!!!! CHECK!!!

print(A.dtype)