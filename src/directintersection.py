import torch
from sparse_reshaper import ints_to_tuples, tuples_to_ints
from directproduct import directproduct

## test methods
from uniform_random_sparse_tensor import uniform_random_sparse_tensor

###################################
######### PyTorch DELETE ##########
###################################

@torch.jit.script
def pytorch_delete(data, args_to_delete):
    """
    *** delete entries of a torch.tensor given indices (to_delete)
    GIVEN : torch.tensor
    GET   : torch.tensor (without deleted entries)
    """
    mask = torch.ones_like(data, dtype=torch.bool)
    mask[args_to_delete] = False
    return data[mask]

###################################
####### DIRECT INTERSECTION #######
###################################

def make_matrix(tensor_in, adj_matrix, side="left"): ## leftside xor rightside, dense xor sparse: make jit......
    """
    *** Fold a Tensor with given external xor internal indices into a external-internal matrix
    """

    internal_index = adj_matrix
    external_index = pytorch_delete( torch.arange(tensor_in.ndim), internal_index)
    shape = torch.tensor(tensor_in.shape)

    if adj_matrix.shape[0]==tensor_in.ndim: ### then reshape to a vector
        s_A = [torch.prod(shape[internal_index]).item()]
        if tensor_in.is_sparse:
            I      = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
            matrix = torch.sparse_coo_tensor( torch.reshape(I,(1,I.shape[0])) , tensor_in._values(), tuple(s_A))
        else:
            permute = [int(elem.item()) for elem in internal_index]
            matrix  = torch.permute(tensor_in, permute).reshape(-1)

    else: ### this creates an internal/external-matrix
        if side=="left":
            s_A = (torch.prod(shape[external_index]).item(), torch.prod(shape[internal_index]).item())
            if tensor_in.is_sparse:
                I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
                E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
                EI   = torch.stack([E, I], axis=0)
                matrix = torch.sparse_coo_tensor( EI , tensor_in._values(), tuple(s_A))

            else: ## then dense
                permute = [int(elem.item()) for elem in torch.concat( [external_index, internal_index] )]
                matrix  = torch.permute(tensor_in, permute).reshape(s_A)

        else: ## external-indices on right-side
            s_A = (torch.prod(shape[internal_index]).item(), torch.prod(shape[external_index]).item())
            if tensor_in.is_sparse:
                I    = tuples_to_ints( tensor_in._indices()[internal_index,:], shape[internal_index] )
                E    = tuples_to_ints( tensor_in._indices()[external_index,:], shape[external_index] )
                IE   = torch.stack([I, E], axis=0)
                matrix = torch.sparse_coo_tensor( IE , tensor_in._values(), tuple(s_A))
            else: ## then dense
                permute = [int(elem.item()) for elem in torch.concat( [internal_index, external_index] )]
                matrix  = torch.permute(tensor_in, permute).reshape(s_A)

    return matrix, shape[external_index]

## TESTS
#M = uniform_random_sparse_tensor(15, torch.tensor([12,4,8]))
#N = make_matrix(M, torch.tensor([1,0]), side="left") ##.to_dense()
#print(N)

###################################
############ TWO TRACE ############
###################################

def twotrace(a, b, adj_matrix):
    if adj_matrix.numel()==0: ## directproduct
        m_c = directproduct(a,b)
    else: ## directintersection
        if a.is_sparse and b.is_sparse: ## m_c is sparse
            m_A, exlabel_A = make_matrix(a, adj_matrix[0], side="left" )
            m_B, exlabel_B = make_matrix(b, adj_matrix[1], side="right")
            m_c = m_A @ m_B ## let it be a vector....
            II  = torch.concat([ints_to_tuples(m_c._indices()[0], exlabel_A), ints_to_tuples(m_c._indices()[1], exlabel_B)])
            m_c = torch.sparse_coo_tensor(II, m_c._values(), [int(i) for i in torch.concat([exlabel_A, exlabel_B])])
        if (not a.is_sparse) and b.is_sparse: ## A==dense & B==Sparse !!!!!!!!!!!!!!!! must put dense matrix to the right!!!!
            m_A, exlabel_A = make_matrix(a, adj_matrix[0], side="right")
            m_B, exlabel_B = make_matrix(b, adj_matrix[1], side="left")
            m_c = (m_B @ m_A).swapaxes(0,1)  ## BA -> AB 
            m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
        if (a.is_sparse) and (not b.is_sparse): ## A==Sparse & B==dense
            m_A, exlabel_A = make_matrix(a, adj_matrix[0], side="left" )
            m_B, exlabel_B = make_matrix(b, adj_matrix[1], side="right")
            m_c = m_A @ m_B ## let it be a vector....
            m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
        if (not a.is_sparse) and (not b.is_sparse): ## A==dense & B==dense
            m_A, exlabel_A = make_matrix(a, adj_matrix[0], side="left" )
            m_B, exlabel_B = make_matrix(b, adj_matrix[1], side="right")
            m_c = m_A @ m_B ## let it be a vector....
            m_c = m_c.reshape([int(i) for i in torch.concat([exlabel_A, exlabel_B])])
    return m_c

## TESTS
#A = uniform_random_sparse_tensor(35, torch.tensor([12,4,8,3]))
#B = uniform_random_sparse_tensor(45, torch.tensor([6,12,8]))
#adjmatrix = torch.tensor([[0,1],[2,2]]).T

#C = twotrace(A, B, adjmatrix)
#D = twotrace(A, B.to_dense(), adjmatrix)
#E = twotrace(A.to_dense(), B, adjmatrix) ### 1st one being dense
#F = twotrace(A.to_dense(), B.to_dense(), adjmatrix)
#print( torch.allclose(C.to_dense(), D), torch.allclose(D, E), torch.allclose(E, F))