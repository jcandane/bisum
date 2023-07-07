####
##
## Here we test the Tr method 
##
####

import numpy as np
import torch
from uniform_random_sparse_tensor import uniform_random_sparse_tensor
from main import bisum

from _randomeinsumstring import random_einsum_string

### einsum-string to ncon....
#print( random_einsum_string() )

def test_pytorch_einsum(label, max_array_size=1.e8, density=0.1):

    rhs = label.split("->")
    if len(rhs)==2:
        lhs = [ np.asarray([ ord(j) for j in i]) for i in rhs[0].split(",")]
    else:
        lhs = [ np.asarray([ ord(j) for j in i]) for i in rhs.split(",")]

    ### get max sizes for axes
    max_dim  = float(np.amax([ len(i) for i in lhs ]))
    max_size = max_array_size**(1 / max_dim)

    ###
    cutoff   = ( len(lhs[0]) )
    uni, inv = np.unique(np.concatenate(lhs), return_inverse=True)
    axes_siz = np.random.randint(3, max_size, size=len(uni))
    sizes    = axes_siz[inv]

    ###
    tensor_shapes = [ sizes[:cutoff] , sizes[cutoff:] ]
    sizes         = [ density*np.prod(tensor_shapes[0]) , density*np.prod(tensor_shapes[1]) ]

    ## create sparse-tensors, get shape from label and pick Number-of-elements
    A = uniform_random_sparse_tensor(150, torch.tensor([12,13,14]))
    B = uniform_random_sparse_tensor(165, torch.tensor([13,13,14]))

    return np.allclose( np.einsum(label, A.to_dense().numpy(), B.to_dense().numpy()), bisum(label, A, B).to_dense().numpy() )

labels = "ijk, jlk -> il"
print( test_pytorch_einsum(labels) )





rhs = labels.split("->")
if len(rhs)==2:
    lhs = [ np.asarray([ ord(j) for j in i]) for i in rhs[0].split(",")]
else:
    lhs = [ np.asarray([ ord(j) for j in i]) for i in rhs.split(",")]
print(lhs)

#print( [ i.split(",") for i in labels.split("->")] ) #.replace(" ","") )
#ad = np.asarray([ ord(i) for i in labels.replace("->","").replace(" ","").replace(",","") ])
#print(ad)
print( np.amax([ len(i) for i in lhs ]) )

uni, inv = np.unique(np.concatenate(lhs), return_inverse=True) #iargsort
cutoff   = ( len(lhs[0]) )


print(inv)
print( np.concatenate(lhs)[np.argsort(np.concatenate(lhs))] )
print(np.arange(100, 100+len(uni))[inv])

print( np.random.randint(3, 100, size=len(uni)) )