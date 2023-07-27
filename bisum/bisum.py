"""
bisum.py

Primary Function of the bisum package, for sparse & dense tensor partial-tracing
"""

import torch

###
from .labels import einsumstr_to_labels, ncon_to_labels
from .tensordot import ss_tensordot , sd_tensordot , sd_tensordot
from .tensordot import ss_tensordot_, sd_tensordot_, ds_tensordot_
from .dense_intra import den_tensor_intraTr, den_post_intraTr, den_post_trans

from .sparse_intra import spa_post_trans, spa_tensor_intraTr, spa_post_intraTr 
from .sparse_intra import spa_post_trans_, spa_tensor_intraTr_, spa_post_intraTr_

def bisum(Rx, a, b):
    """

    """
    if torch.is_tensor(Rx): # and Rx.shape[0]==2:  # is adj.matrix (no post transpose nor slice) 
        if (not a.is_sparse) and (not b.is_sparse): ## both dense
            c = torch.tensordot(a, b, dims=Rx)
        else:
            if (a.is_sparse) and (b.is_sparse):
                c = ss_tensordot(a, b, dims=Rx)
            else:
                c = sd_tensordot(a, b, dims=Rx)

    else:
        if isinstance(Rx, list): # ncon  (no post transpose)  
            LHS, RHS, lhs, rhs, inTr, adjmat = ncon_to_labels(Rx)
        else:
            if isinstance(Rx, str): # einsum ncon_to_labels
                LHS, RHS, lhs, rhs, inTr, adjmat = einsumstr_to_labels(Rx, device=a.device)
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
                
                a_index, a_data, a_shape = spa_tensor_intraTr_(a, LHS[0], inTr[0])
                b_index, b_data, b_shape = spa_tensor_intraTr_(b, LHS[1], inTr[1])
                c_index, c_data, c_shape = ss_tensordot_(a_index, a_data, a_shape, b_index, b_data, b_shape, dims=adjmat)
                c_index, c_data, c_shape = spa_post_intraTr_(c_index, c_data, c_shape, rhs)
                c = spa_post_trans_(c_index, c_data, c_shape, rhs, RHS)
                
            else:
                if (a.is_sparse) and (not b.is_sparse):
                    a_index, a_data, a_shape = spa_tensor_intraTr_(a, LHS[0], inTr[0])
                    #a = spa_tensor_intraTr(a, LHS[0], inTr[0]) ##!!!
                    b = den_tensor_intraTr(b, LHS[1], inTr[1])

                    #c = sd_tensordot(a, b, dims=adjmat) ##!!!
                    c = sd_tensordot_(a_index, a_data, a_shape, b, dims=adjmat)

                    c = den_post_intraTr(c, rhs)
                    c = den_post_trans(c, rhs, RHS)

                else: ## a is dense and b is sparse
                    a = den_tensor_intraTr(a, LHS[0], inTr[0])
                    #b = spa_tensor_intraTr(b, LHS[1], inTr[1]) ##!!!
                    b_index, b_data, b_shape = spa_tensor_intraTr_(b, LHS[1], inTr[1])

                    #c = sd_tensordot(a, b, dims=adjmat) ##!!!
                    c = ds_tensordot_(a, b_index, b_data, b_shape, dims=adjmat)
                    
                    c = den_post_intraTr(c, rhs)
                    c = den_post_trans(c, rhs, RHS)
    return c