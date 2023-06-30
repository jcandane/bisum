import torch
from sparse_reshaper import ints_to_tuples, tuples_to_ints

### random-connectivity is related to sequences of random-ordered-integers
@torch.jit.script
def random_unique_ints(amount : int, limit : int, dtype : torch.dtype = torch.int32): ## both amount & limit should be arrays
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

    output=torch.zeros(0, dtype=dtype)
    leftover=1*amount
    while leftover>0:
        output   = torch.concatenate([ torch.randint(0, limit, [int(leftover)], dtype=dtype), output ])
        output   = torch.unique(output)
        leftover = amount-int(output.shape[0])
    return output

@torch.jit.script
def random_sparse_tensor(N, shape):
    Random_tuple = random_unique_ints(N, torch.prod(shape), dtype=torch.int32)
    return ints_to_tuples(Random_tuple, shape).T, torch.rand(N), shape

def uniform_random_sparse_tensor(N, shape):
    Random_tuples = ints_to_tuples(random_unique_ints(N, torch.prod(shape), dtype=torch.int32), shape)
    return torch.sparse_coo_tensor(Random_tuples, torch.rand(N), tuple(shape))

### TESTS
#a     = uniform_random_sparse_tensor(15000, torch.tensor([4,4,4,4,4,4,4,4]))
#print(a)