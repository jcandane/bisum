import torch
from _basicfunctions import ints_to_tuples

@torch.jit.script
def random_unique_ints(amount : int, limit : int, dtype : torch.dtype = torch.int32, device : torch.device=torch.device("cpu")):
    """
    *June/2023/jcandanedo
    *iteratively-sampling until the 'amount' is reached
    *random-connectivity is related to sequences of random-ordered-integers
    *
    GIVEN : amount (number of integers desired)
            limit (MAX value of integers, MIN==0)
    GET   : ouput (1d-torch.tesnor of unique-sorted integers from range [0, MAX) )
    ⚠ amount <= limit
    ⚠ this while loop is slow, but happens slow to once....
    ⚠ expected number of duplicates, could help speed this up
    *** Limit max/min value of limit
    """
    if amount>limit:
        raise ValueError("amount <= limit")

    output=torch.zeros(0, dtype=dtype, device=device)
    leftover=1*amount
    while leftover>0:
        output   = torch.concatenate([ torch.randint(0, limit, [int(leftover)], dtype=dtype, device=device), output ])
        output   = torch.unique(output)
        leftover = amount-int(output.shape[0])
    return output

@torch.jit.script
def random_sparse_tensor(N, shape, device : torch.device=torch.device("cpu")):
    Random_tuple = random_unique_ints(N, torch.prod(shape), dtype=torch.int32, device=device)
    return ints_to_tuples(Random_tuple, shape).T, torch.rand(N), shape

#@torch.jit.script
def uniform_random_sparse_tensor(N : int, shape, dtype : torch.dtype = torch.float32, indexdtype : torch.dtype = torch.int32, device : torch.device=torch.device("cpu")):
    Random_tuples = ints_to_tuples(random_unique_ints(N, torch.prod(shape), dtype=indexdtype, device=device), shape)
    return torch.sparse_coo_tensor(Random_tuples, torch.rand(N, dtype=dtype, device=device), [int( s.item() ) for s in shape] ) ## tuple(shape)

### TESTS torch.tensor(42, dtype=torch.int32)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#b     = random_unique_ints(15, 246)#
#a     = uniform_random_sparse_tensor(15000, torch.tensor([4,4,4,4,4,4,4,4], device=device), device=device)
#print(a._indices().dtype)