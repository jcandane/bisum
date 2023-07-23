"""
dev_defs.py

Specific Functions used for this package.
"""

import torch
from generic_functions import pytorch_delete, tuples_to_ints, first_occurrence_mask, ints_to_tuples
from directproduct import directproduct
from typing import List

@torch.jit.script
def sp_tensor_to_matrix(tensor_in, adj_matrix, left : bool = True):
    """ !!!! tensor_in should instead be:    indices, data, shape (as tensors), because by DEAFULT SparseTensors change index types to int64 :(
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

    #print("--------------")
    #print(shape[external_index], internal_index, external_index, shape)
    return matrix, shape[external_index]

@torch.jit.script
def dn_tensor_to_matrix(tensor_in, adj_matrix, left : bool = True):
    """ !!!! tensor_in should instead be:    indices, data, shape (as tensors), because by DEAFULT SparseTensors change index types to int64 :(
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

@torch.jit.script
def sdtensordot(a, b, dims=None):
    """
    GIVEN : a,b (torch.tensor{sparse})
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{dense} XOR torch.tensor{sparse} is just direct-product)
    """
    if dims.numel()==0: ## directproduct
        c = directproduct(a,b)
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
def sptensordot(a, b, dims=None):
    """
    GIVEN : a,b (torch.tensor{sparse})
            dims (2d-int-torch.tensor, with 0th axis being 2: e.g. shape=(2,4))
    GET   : c (torch.tensor{sparse})
    """
    if dims.numel()==0: ## directproduct
        c = directproduct(a,b)
    else: ## directintersection
        m_A, exlabel_A = sp_tensor_to_matrix(a, dims[0], left=True ) ## True
        m_B, exlabel_B = sp_tensor_to_matrix(b, dims[1], left=False ) ## False

        if torch.numel(exlabel_A)==0 and torch.numel(exlabel_B)!=0:
            c = m_A @ m_B ## let it be a....
            I = ints_to_tuples(c._indices()[1], exlabel_B)
            c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in exlabel_B])
        else:
            if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)!=0:
                c = m_A @ m_B   ## let it be a....
                I = ints_to_tuples(c._indices()[0], exlabel_A)
                c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in exlabel_A])
            else:
                if torch.numel(exlabel_B)!=0 and torch.numel(exlabel_A)!=0:
                    c = m_A @ m_B ## let it be a....
                    I = torch.concat([ints_to_tuples(c._indices()[0], exlabel_A), ints_to_tuples(c._indices()[1], exlabel_B)])
                    c = torch.sparse_coo_tensor(I, c._values(), [int(i) for i in torch.concat([exlabel_A, exlabel_B])])
                else:
                    if torch.numel(exlabel_B)==0 and torch.numel(exlabel_A)==0:
                        c = m_A @ m_B ## let it be a....
                    else:
                        raise ValueError
    return c

### TEST
#from uniform_random_sparse_tensor import uniform_random_sparse_tensor

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#print(device)

#A  = uniform_random_sparse_tensor(150, torch.tensor([15, 15, 15]), device=device)
#B  = uniform_random_sparse_tensor( 14, torch.tensor([15, 15, 15]), device=device)
#Rx = torch.tensor([[0],[1]], device=device)
#Rx = torch.tensor([[1,0],[2,1]], device=device)

#c  = torch.tensordot(A.to_dense(), B.to_dense(), dims=Rx)
#c_s= sdtensordot(A, B.to_dense(), dims=Rx)
#c_q= sdtensordot(A.to_dense(), B, dims=Rx)
#print(torch.allclose( c, c_s ) == torch.allclose( c, c_q ))

#A  = uniform_random_sparse_tensor(150, torch.tensor([18, 15, 17]), device=device)
#B  = uniform_random_sparse_tensor( 14, torch.tensor([12, 15, 15]), device=device)
#Rx = torch.tensor([[1,1],[2,1]], device=device)
#c_s= sdtensordot(A, B, dims=Rx)
#print(c_s.shape)

##
## EINSTEIN-STRING
##

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