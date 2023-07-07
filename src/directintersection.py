import torch
from directproduct import directproduct
from _basicfunctions import ints_to_tuples, tuples_to_ints, pytorch_delete

## test methods
from uniform_random_sparse_tensor import uniform_random_sparse_tensor

###################################
####### DIRECT INTERSECTION #######
################################### 

@torch.jit.script
def make_matrix(tensor_in, adj_matrix, left : bool = True):
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
    #external_index = pytorch_delete( torch.arange(tensor_in.ndim), internal_index) ### !!!arange not in Everything not included....
    external_index = pytorch_delete( arange , internal_index) ### !!!arange not in Everything not included....
    shape = torch.concat(shaper) #torch.tensor(tensor_in.shape) ### !!!arange not in 
    
    if adj_matrix.shape[0]==tensor_in.ndim: ### then reshape to a vector, no external-indices
        sA = [1, torch.prod(shape[internal_index]).item(), 1]
        sA = [int( s ) for s in sA]
        if tensor_in.is_sparse:
            I      = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
            matrix = torch.sparse_coo_tensor( torch.reshape(I,(1,I.shape[0])) , tensor_in._values(), sA )
        else:
            permute = [int(elem.item()) for elem in internal_index]
            matrix  = torch.permute(tensor_in, permute).reshape(-1)

    else: ### this creates an internal/external-matrix
        if left:
            sA = torch.concat([torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1), torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            if tensor_in.is_sparse:
                I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] ) ### check devices for shape & tensor_in
                E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
                EI   = torch.stack([E, I], dim=0)
                matrix = torch.sparse_coo_tensor( EI , tensor_in._values(), sA )

            else: ## then dense
                permute = [int(elem.item()) for elem in torch.concat( [external_index, internal_index] )]
                matrix  = torch.permute(tensor_in, permute).reshape(sA)

        else: ## external-indices on right-side
            sA = torch.concat([torch.prod(shape[internal_index].reshape((1,internal_index.shape[0])), dim=1), torch.prod(shape[external_index].reshape((1,external_index.shape[0])), dim=1)])
            sA = [int( s.item() ) for s in sA]
            if tensor_in.is_sparse:
                I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
                E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
                IE   = torch.stack([I, E], dim=0)
                matrix = torch.sparse_coo_tensor( IE , tensor_in._values(), sA )
            else: ## then dense
                permute = [int(elem.item()) for elem in torch.concat( [internal_index, external_index] )]
                matrix  = torch.permute(tensor_in, permute).reshape( sA )

    return matrix, shape[external_index]

## TESTS 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#print(device)
#M = uniform_random_sparse_tensor(15, torch.tensor([12,4,8]), device=device, indexdtype=torch.int32)
#print(M._indices().dtype)
#N = make_matrix(M, torch.tensor([1,0], device=device), left=True) ##.to_dense()
#print(N)

###################################
############ TWO TRACE ############
###################################

@torch.jit.script
def twotrace(a, b, adj_matrix):
    if adj_matrix.numel()==0: ## directproduct
        m_c = directproduct(a,b)
    else: ## directintersection
        if a.is_sparse and b.is_sparse: ## m_c is sparse
            m_A, exlabel_A = make_matrix(a, adj_matrix[0], left=True )
            m_B, exlabel_B = make_matrix(b, adj_matrix[1], left=False)
            m_c = m_A @ m_B ## let it be a vector....
            II  = torch.concat([ints_to_tuples(m_c._indices()[0], exlabel_A), ints_to_tuples(m_c._indices()[1], exlabel_B)])
            m_c = torch.sparse_coo_tensor(II, m_c._values(), [int(i) for i in torch.concat([exlabel_A, exlabel_B])])
        else:
            if (not a.is_sparse) and b.is_sparse: ## A==dense & B==Sparse, must place dense matrix to the right!
                m_A, exlabel_A = make_matrix(a, adj_matrix[0], left=False)
                m_B, exlabel_B = make_matrix(b, adj_matrix[1], left=True )
                m_c = (m_B @ m_A).swapaxes(0,1)  ## BA -> AB 
                m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
            else:
                if (a.is_sparse) and (not b.is_sparse): ## A==Sparse & B==dense
                    m_A, exlabel_A = make_matrix(a, adj_matrix[0], left=True )
                    m_B, exlabel_B = make_matrix(b, adj_matrix[1], left=False)
                    m_c = m_A @ m_B ## let it be a vector....
                    m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
                else: ## A==dense & B==dense
                    m_A, exlabel_A = make_matrix(a, adj_matrix[0], left=True )
                    m_B, exlabel_B = make_matrix(b, adj_matrix[1], left=False)
                    m_c = m_A @ m_B ## let it be a vector....
                    m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
    return m_c

## Quick TESTS
#A = uniform_random_sparse_tensor(35, torch.tensor([12,4,8,3]))
#B = uniform_random_sparse_tensor(45, torch.tensor([6,12,8]))
#adjmatrix = torch.tensor([[0,1],[2,2]]).T

#C = twotrace(A, B, adjmatrix)
#D = twotrace(A, B.to_dense(), adjmatrix)
#E = twotrace(A.to_dense(), B, adjmatrix) ### 1st one being dense
#F = twotrace(A.to_dense(), B.to_dense(), adjmatrix)
#print( torch.allclose(C.to_dense(), D), torch.allclose(D, E), torch.allclose(E, F))