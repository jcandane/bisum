"""
tensordot.py

This generalizes tensordot for sparse tensors (with dense tensors).
"""

import torch
from generic_functions import ints_to_tuples, cartesian_product
from tensor_to_matrix import sp_tensor_to_matrix, dn_tensor_to_matrix, rawsp_tensor_to_matrix

@torch.jit.script
def sd_outer(a, b):
    """
    GIVEN : a (sparse)
            b (dense)
    GET   : c   (torch.tensor{dense})
    """
    if a.is_sparse:
        a=a.to_dense()
    else:
        if b.is_sparse:
            b=b.to_dense()
        else:
            torch.tensordot(a, b, dims=(0))
    return  torch.tensordot(a, b, dims=(0))

@torch.jit.script
def sd_outer_(a_index, a_data, a_shape, b):
    """
    GIVEN : a (sparse)
            b (dense)
    GET   : c   (torch.tensor{dense})
    """
    a = torch.sparse_coo_tensor(a_index, a_data, [int(q.item()) for q in a_shape])
    a=a.to_dense()
    return  torch.tensordot(a, b, dims=(0))

@torch.jit.script
def ds_outer_(a, b_index, b_data, b_shape):
    """
    GIVEN : a (sparse)
            b (dense)
    GET   : c   (torch.tensor{dense})
    """
    b = torch.sparse_coo_tensor(b_index, b_data, [int(q.item()) for q in b_shape])
    b=b.to_dense()
    return torch.tensordot(a, b, dims=(0))

@torch.jit.script
def sparse_outer(a,b):
    """
    GIVEN : a,b (torch.sparse_coo_tensor)
    GET   : c   (torch.sparse_coo_tensor)
    """
    data = torch.outer(a._values(), b._values()).reshape(-1)
    inde = cartesian_product(a._indices(),b._indices())
    size = torch.concat((torch.tensor(a.shape, device=a.device),torch.tensor(b.shape, device=b.device)))
    size = [int(elem.item()) for elem in size]
    return torch.sparse_coo_tensor(inde, data, size) #return [inde, data, size]

@torch.jit.script
def sd_tensordot(a, b, dims=None):
    """
    GIVEN : a, sparse torch.tensor
            b, dense  torch.tensor
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{dense} XOR torch.tensor{sparse} is just direct-product)
    """
    if dims.numel()==0: ## directproduct
        c = sd_outer(a, b)
    else: ## directintersection
        if (not a.is_sparse) and (b.is_sparse): # a is dense && b is sparse
            m_A, exlabel_A = dn_tensor_to_matrix(a, dims[0], left=False ) ## True
            m_B, exlabel_B = sp_tensor_to_matrix(b, dims[1], left=True ) ## False
            c = m_B @ m_A ### SPARSE is first??!?
            c = c.T.reshape( [ int(i) for i in torch.concat([torch.flatten(exlabel_B), torch.flatten(exlabel_A)])])
        else: ## b is dense
            m_A, exlabel_A = sp_tensor_to_matrix(a, dims[0], left=True ) ## True
            m_B, exlabel_B = dn_tensor_to_matrix(b, dims[1], left=False  ) ## False
            c = m_A @ m_B ## Sparse is 1st
            c = c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
    return c

@torch.jit.script
def ds_tensordot(a, b, dims=None):
    """
    GIVEN : a, dense  torch.tensor
            b, sparse torch.tensor
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{dense} XOR torch.tensor{sparse} is just direct-product)
    """
    if dims.numel()==0: ## directproduct
        c = sd_outer(a, b)
    else: ## directintersection
        if (not a.is_sparse) and (b.is_sparse): # a is dense && b is sparse
            m_A, exlabel_A = dn_tensor_to_matrix(a, dims[0], left=False ) ## True
            m_B, exlabel_B = sp_tensor_to_matrix(b, dims[1], left=True ) ## False
            c = m_B @ m_A ### SPARSE is first??!?
            c = c.T.reshape( [ int(i) for i in torch.concat([torch.flatten(exlabel_B), torch.flatten(exlabel_A)])])
        else: ## b is dense
            m_A, exlabel_A = sp_tensor_to_matrix(a, dims[0], left=True ) ## True
            m_B, exlabel_B = dn_tensor_to_matrix(b, dims[1], left=False  ) ## False
            c = m_A @ m_B ## Sparse is 1st
            c = c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
    return c

@torch.jit.script
def ss_tensordot(a, b, dims=None):
    """
    GIVEN : a,b (torch.tensor{sparse})
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{sparse})
    """
    if dims.numel()==0: ## directproduct 
        c = sparse_outer(a,b)
    else: ## directintersection
        m_A, exlabel_A = sp_tensor_to_matrix(a, dims[0], left=True ) ## True
        m_B, exlabel_B = sp_tensor_to_matrix(b, dims[1], left=False ) ## False

        if torch.numel(exlabel_A)==0 and torch.numel(exlabel_B)!=0:
            c = m_A @ m_B ## let it be a....
            I = ints_to_tuples(c._indices()[1], exlabel_B)
            c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in exlabel_B])
        else:
            if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)!=0:
                c = m_A @ m_B   ## B side is null, result is a "vector" in A
                I = ints_to_tuples(c._indices()[0], exlabel_A)
                c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in exlabel_A])
            else:
                if torch.numel(exlabel_B)!=0 and torch.numel(exlabel_A)!=0:
                    c = m_A @ m_B ## A side is null, result is a "vector" in B
                    I = torch.concat([ints_to_tuples(c._indices()[0], exlabel_A), ints_to_tuples(c._indices()[1], exlabel_B)])
                    c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in torch.concat([exlabel_A, exlabel_B])])
                else: ## A & B side is null, result is scalar, a trivial sparse tensor
                    if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)==0:
                        c = m_A @ m_B ## let it be a....
                    else:
                        raise ValueError
    return c

@torch.jit.script
def ss_tensordot_(a_index, a_data, a_shape, b_index, b_data, b_shape, dims=None):
    """
    GIVEN : a,b (torch.tensor{sparse})
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{sparse})
    """
    if dims.numel()==0: ## directproduct
        I     = cartesian_product( a_index, b_index )
        data  = torch.outer( a_data, b_data ).reshape(-1)
        shape_= torch.concat([a_shape, b_shape])
    else: ## directintersection
        m_A, exlabel_A = rawsp_tensor_to_matrix(a_index, a_data, a_shape, dims[0], left=True )
        m_B, exlabel_B = rawsp_tensor_to_matrix(b_index, b_data, b_shape, dims[1], left=False)

        if torch.numel(exlabel_A)==0 and torch.numel(exlabel_B)!=0:
            c = m_A @ m_B ## let it be a....
            I = ints_to_tuples(c._indices()[1], exlabel_B)
            data   = c._values()
            shape_ = exlabel_B
        else:
            if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)!=0:
                c = m_A @ m_B   ## B side is null, result is a "vector" in A
                I = ints_to_tuples(c._indices()[0], exlabel_A)
                data   = c._values()
                shape_ = exlabel_A
            else:
                if torch.numel(exlabel_B)!=0 and torch.numel(exlabel_A)!=0:
                    c = m_A @ m_B ## A side is null, result is a "vector" in B
                    I = torch.concat([ints_to_tuples(c._indices()[0], exlabel_A), ints_to_tuples(c._indices()[1], exlabel_B)])
                    data   = c._values()
                    shape_ = torch.concat([exlabel_A, exlabel_B])
                else: ## A & B side is null, result is scalar, a trivial sparse tensor
                    if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)==0:
                        c = m_A @ m_B ## let it be a....
                        I      = torch.zeros_like( torch.unsqueeze(a_index[:,0], 0)[:,0] , dtype=a_index.dtype)
                        data   = c._values()
                        shape_ = torch.ones_like( torch.unsqueeze(a_index[:,0], 0)[:,0] , dtype=a_shape.dtype) 
                    else:
                        raise ValueError
    return I, data, shape_

@torch.jit.script
def ds_tensordot_(a, b_index, b_data, b_shape, dims=None):
    """
    GIVEN : a, dense  torch.tensor
            b, sparse torch.tensor
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{dense} XOR torch.tensor{sparse} is just direct-product)
    """
    if dims.numel()==0: ## directproduct
        c = ds_outer_(a, b_index, b_data, b_shape)
    else: ## directintersection
        # a is dense && b is sparse
        m_A, exlabel_A = dn_tensor_to_matrix(a, dims[0], left=False ) ## True
        m_B, exlabel_B = rawsp_tensor_to_matrix(b_index, b_data, b_shape, dims[1], left=False)
        
        #m_B, exlabel_B = sp_tensor_to_matrix(b, dims[1], left=True ) ## False
        c = m_B @ m_A ### SPARSE is first??!?
        c = c.T.reshape( [ int(i) for i in torch.concat([torch.flatten(exlabel_B), torch.flatten(exlabel_A)])])
    return c

@torch.jit.script
def sd_tensordot_(a_index, a_data, a_shape, b, dims=None):
    """
    GIVEN : a, sparse torch.tensor
            b, dense  torch.tensor
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{dense} XOR torch.tensor{sparse} is just direct-product)
    """
    if dims.numel()==0: ## directproduct
        c = sd_outer_(a_index, a_data, a_shape, b)
    else: ## directintersection
        m_A, exlabel_A = rawsp_tensor_to_matrix(a_index, a_data, a_shape, dims[0], left=True ) ## True
        m_B, exlabel_B = dn_tensor_to_matrix(b, dims[1], left=False  ) ## False
        c = m_A @ m_B ## Sparse is 1st
        c = c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
    return c