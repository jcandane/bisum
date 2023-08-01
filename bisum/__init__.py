"""
bisum -- efficient sparse tensor contractions
=============================================

The :mod:`bisum` package implements sparse-tensor partial-trace operations with `PyTorch`_.
The primary function for users is :func:`bisum`.

This program traces 2 sparse-tensor (:class:`torch.tensor` objects) via 3 Tracing-Prescription:

1. ``{einsum}`` string (like numpy, str, labelling each tensor axis)
2. ``ncon`` (used in the tensor-network community, list of 1d int :class:`torch.tensor`, labelling each tensor axis)
3. adjacency-matrix (as in :func:`numpy.tensordot`, ``(2,n)`` 2d int torch.tensor, with n being the number of indices identified between the two tensors)

.. _`PyTorch`: https://pytorch.org/


Quickstart
----------

Let's begin by initializing the 2 tensors, we can initialize random-sparse-tensors::

    import torch
    from bisum import bisum

    shape_A = torch.tensor([8,7,7,4,11,6])
    shape_B = torch.tensor([9,7,3,7,11,8])
    A = torch.rand(shape_A)
    B = torch.rand(shape_B)

Suppose we would like to compute the following partial-trace/tensor-contraction :math:`C_{njwl} = A_{iksndj} B_{wklsdi}`::

    C_einsum = bisum("iksndj, wklsdi -> njwl", A, B)
    C_ncon   = bisum([[-1,-2,-3,4,-5,6],[1,-2,3,-3,-5,-1]], A, B)
    C_adjmat = bisum(torch.tensor([[0,1,2,4],[5,1,3,4]]), A, B)

    print(torch.allclose(C_einsum, C_ncon) and torch.allclose(C_ncon, C_adjmat))


while the pure tensor-product, :math:`\otimes` is::

    import numpy as np

    C_einsum = bisum("abcdef, ghijkl", A, B)
    C_ncon   = bisum([], A, B)
    C_adjmat = bisum(torch.tensor([]), A, B)

    print(np.allclose(C_einsum, C_ncon) and np.allclose(C_ncon, C_adjmat))

"""

from .bisum import bisum

__all__ = ["bisum"]
