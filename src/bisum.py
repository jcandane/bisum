"""
bisum.py

Primary Function of the bisum package, for sparse & dense tensor partial-tracing
"""

import torch

###
from dev_defs import einsumstr_to_labels, ncon_to_labels, sdtensordot, sptensordot
from sparse_intra import spa_post_trans, spa_tensor_intraTr, spa_post_intraTr
from dense_intra import den_tensor_intraTr, den_post_intraTr, den_post_trans

def bisum(Rx, a, b):
    """

    """
    if torch.is_tensor(Rx): # and Rx.shape[0]==2:  # is adj.matrix (no post transpose nor slice) 
        if (not a.is_sparse) and (not b.is_sparse): ## both dense
            c = torch.tensordot(a, b, dims=Rx)
        else:
            if (a.is_sparse) and (b.is_sparse):
                c = sptensordot(a, b, dims=Rx)
            else:
                c = sdtensordot(a, b, dims=Rx)

    else:
        if isinstance(Rx, list): # ncon  (no post transpose)  
            LHS, RHS, lhs, rhs, inTr, adjmat = ncon_to_labels(Rx)
        else:
            if isinstance(Rx, str): # einsum ncon_to_labels
                LHS, RHS, lhs, rhs, inTr, adjmat = einsumstr_to_labels(Rx)
            else:
                raise ValueError("tracing instructions are not valid")

        if (not a.is_sparse) and (not b.is_sparse): ## both dense
            a = den_tensor_intraTr(a, LHS[0], inTr[0])
            b = den_tensor_intraTr(b, LHS[1], inTr[1])

            c = torch.tensordot(a, b, dims=adjmat)

            c = den_post_intraTr(c, rhs)
            c = den_post_trans(c, rhs, RHS) #for dense
        else:
            if a.is_sparse and b.is_sparse: ## both sparse
                a = spa_tensor_intraTr(a, LHS[0], inTr[0])
                #print(a.shape, inTr[1]) ### !!!
                b = spa_tensor_intraTr(b, LHS[1], inTr[1])

                c = sptensordot(a, b, dims=adjmat)

                c = spa_post_intraTr(c, rhs) ## concat lhs removing internals!!!
                c = spa_post_trans(c, rhs, RHS)
                
            else:
                if (a.is_sparse) and (not b.is_sparse):
                    a = spa_tensor_intraTr(a, LHS[0], inTr[0])
                    b = den_tensor_intraTr(b, LHS[1], inTr[1])

                    c = sdtensordot(a, b, dims=adjmat)

                    c = den_post_intraTr(c, rhs)
                    c = den_post_trans(c, rhs, RHS)

                else: ## a is dense and b is sparse
                    a = den_tensor_intraTr(a, LHS[0], inTr[0])
                    b = spa_tensor_intraTr(b, LHS[1], inTr[1])

                    c = sdtensordot(a, b, dims=adjmat)
                    
                    c = den_post_intraTr(c, rhs)
                    c = den_post_trans(c, rhs, RHS)
    return c