# bisum
PyTorch Sparse-Tensor Partial-Trace (PyTorch_sTr)
This program traces 2-sparse-tensor (torch.tensor objects) via 3 Tracing-Prescription:
1. einsum string (like numpy, str, labelling each tensor axis)
2. ncon (used in the tensor-network community, list of 1d int torch.tensor, labelling each tensor axis)
3. adjacency-matrix (as in numpy.tensordot, (2,n) 2d int torch.tensor, with n being the number of indices idenified between the two tensors)
