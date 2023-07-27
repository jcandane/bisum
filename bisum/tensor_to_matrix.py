"""
tensor_to_matrix.py

Fold dense and sparse tensors into matrices (dense or sparse)
"""

import torch
from .generic_functions import pytorch_delete, tuples_to_ints


@torch.jit.script
def rawsp_tensor_to_matrix(index, data, shape_, adj_matrix, left : bool = True):
    """ !!!! tensor_in should instead be:    indices, data, shape (as tensors), because by DEAFULT SparseTensors change index types to int64 :(
    *** Fold a Tensor with given external xor internal indices into a external-internal matrix
    GIVEN : index (2d-int-torch.tensor)
            data (1d-torch.tensor)
            shape_ (1d-int-torch.tensor)
            adj_matrix (2d-int-torch.tensor)
            *left (side in-which external-indices will be placed on)
    GET   : matrix (torch.tensor[2d-dense])
            externals-shape (1d-int-torch.tensor, prefolding)
    """

    shaper = [ shape_[i] + 0*adj_matrix[0].reshape((1,1)) for i in range(len(shape_)) ]
    arange = [ adj_matrix[0].reshape((1,1)) for i in range(len(shape_)) ]
    arange = ( torch.cumsum( (1+0*torch.concat(arange))[:,0] , 0 ) - 1 ) ## arange

    internal_index = adj_matrix
    external_index = pytorch_delete( arange , internal_index) ### !!!arange not in Everything not included....
    shape = torch.flatten(torch.concat(shaper)) #!!! #torch.tensor(tensor_in.shape) ### !!!arange not in 
    
    ### LEFT side vector!!  need RIGHT side too!!!!
    if adj_matrix.shape[0]==index.shape[0]: ### then reshape to a vector, no external-indices
        if left:
            sA = [ 1, torch.prod(shape[internal_index]).item() ]
            sA = [int( s ) for s in sA]
            I  = tuples_to_ints( index[internal_index,:], shape[internal_index] )
            I0 = torch.zeros_like( I , dtype=I.dtype)
            I  = torch.stack((I0, I))
            matrix = torch.sparse_coo_tensor( I , data, sA )
        else:
            sA = [ torch.prod(shape[internal_index]).item(), 1] #[1, ... , 1]
            sA = [int( s ) for s in sA]
            I  = tuples_to_ints( index[internal_index,:], shape[internal_index] )
            I0 = torch.zeros_like( I , dtype=I.dtype)
            I  = torch.stack((I, I0))
            matrix = torch.sparse_coo_tensor( I , data, sA )

    else: ### this creates an internal/external-matrix
        if left:
            sA = torch.concat([torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1), torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            I    = tuples_to_ints( index[internal_index,:], shape[internal_index] ) ### check devices for shape & tensor_in
            E    = tuples_to_ints( index[external_index,:], shape[external_index] )
            EI   = torch.stack([E, I], dim=0)
            matrix = torch.sparse_coo_tensor( EI , data, sA )

        else: ## external-indices on right-side
            sA = torch.concat([torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1), torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            I    = tuples_to_ints( index[internal_index,:], shape[internal_index] )
            E    = tuples_to_ints( index[external_index,:], shape[external_index] )
            IE   = torch.stack([I, E], dim=0)
            matrix = torch.sparse_coo_tensor( IE , data, sA )

    return matrix, shape[external_index]


@torch.jit.script
def sp_tensor_to_matrix(tensor_in, adj_matrix, left : bool = True):
    """
    *** Fold a Tensor with given external xor internal indices into a external-internal matrix
    GIVEN : tensor_in (torch.tensor{sparse})
            adj_matrix (2d-int-torch.tensor)
            *left (side in-which external-indices will be placed on)
    GET   : matrix (torch.tensor[2d-dense])
            externals-shape (1d-int-torch.tensor, prefolding)
    """

    shaper = [ tensor_in.shape[i] + 0*adj_matrix[0].reshape((1,1)) for i in range(len(tensor_in.shape)) ]
    arange = [ adj_matrix[0].reshape((1,1)) for i in range(len(tensor_in.shape)) ]
    arange = ( torch.cumsum( (1+0*torch.concat(arange))[:,0] , 0 ) - 1 ) ## arange

    internal_index = adj_matrix
    external_index = pytorch_delete( arange , internal_index) ### !!!arange not in Everything not included....
    shape = torch.flatten(torch.concat(shaper)) #!!! #torch.tensor(tensor_in.shape) ### !!!arange not in 
    
    ### LEFT side vector!!  need RIGHT side too!!!!
    if adj_matrix.shape[0]==tensor_in.ndim: ### then reshape to a vector, no external-indices
        if left:
            sA = [ 1, torch.prod(shape[internal_index]).item() ]
            sA = [int( s ) for s in sA]
            I  = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
            I0 = torch.zeros_like( I , dtype=I.dtype)
            I  = torch.stack((I0, I))
            matrix = torch.sparse_coo_tensor( I , tensor_in._values(), sA )
        else:
            sA = [ torch.prod(shape[internal_index]).item(), 1] #[1, ... , 1]
            sA = [int( s ) for s in sA]
            I  = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
            I0 = torch.zeros_like( I , dtype=I.dtype)
            I  = torch.stack((I, I0))
            matrix = torch.sparse_coo_tensor( I , tensor_in._values(), sA )

    else: ### this creates an internal/external-matrix
        if left:
            sA = torch.concat([torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1), torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] ) ### check devices for shape & tensor_in
            E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
            EI   = torch.stack([E, I], dim=0)
            matrix = torch.sparse_coo_tensor( EI , tensor_in._values(), sA )

        else: ## external-indices on right-side
            sA = torch.concat([torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1), torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
            E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
            IE   = torch.stack([I, E], dim=0)
            matrix = torch.sparse_coo_tensor( IE , tensor_in._values(), sA )

    return matrix, shape[external_index]

@torch.jit.script
def dn_tensor_to_matrix(tensor_in, adj_matrix, left : bool = True):
    """
    *** Fold a Tensor with given external xor internal indices into a external-internal matrix
    GIVEN : tensor_in (torch.tensor)
            adj_matrix (2d-int-torch.tensor)
            *left (side in-which external-indices will be placed on)
    GET   : matrix (torch.tensor[2d-dense])
            externals-shape (1d-int-torch.tensor, prefolding)
    """

    shaper = [ tensor_in.shape[i] + 0*adj_matrix[0].reshape((1,1)) for i in range(len(tensor_in.shape)) ]
    arange = [ adj_matrix[0].reshape((1,1)) for i in range(len(tensor_in.shape)) ]
    arange =   torch.cumsum( (1+0*torch.concat(arange))[:,0] , 0 ) - 1 ## arange

    internal_index = adj_matrix
    external_index = pytorch_delete( arange , internal_index) ### !!!arange not in Everything not included....
    shape = torch.concat(shaper) #torch.tensor(tensor_in.shape) ### !!!arange not in 
    
    if adj_matrix.shape[0]==tensor_in.ndim: ### then reshape to a vector, no external-indices
        if left:
            sA = [ torch.prod(shape[internal_index]).item(), 1]
            sA = [int( s ) for s in sA]
            permute = [int(elem.item()) for elem in internal_index]
            matrix  = torch.permute(tensor_in, permute).reshape(-1)
        else:
            sA = [1, torch.prod(shape[internal_index]).item()]
            sA = [int( s ) for s in sA]
            permute = [int(elem.item()) for elem in internal_index]
            matrix  = torch.permute(tensor_in, permute).reshape(-1)

    else: ### this creates an internal/external-matrix
        if left:
            sA = torch.concat([torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1), torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            permute = [int(elem.item()) for elem in torch.concat( [external_index, internal_index] )]
            matrix  = torch.permute(tensor_in, permute).reshape(sA)

        else: ## external-indices on right-side
            sA = torch.concat([torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1), torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            permute = [int(elem.item()) for elem in torch.concat( [internal_index, external_index] )]
            matrix  = torch.permute(tensor_in, permute).reshape( sA )

    return matrix, shape[external_index]