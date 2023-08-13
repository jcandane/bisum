---
title: 'Sparse `einsum`, `bisum`'
tags:
  - Python
  - PyTorch
  - Sparse-Tensor
  - Tensor
  - Partial-Trace
  - Contraction
authors:
  - name: Julio J. Candanedo
    orcid: 0000-0002-7600-138X
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Department of Physics, Arizona State University, Tempe, AZ 85287, USA
   index: 1
date: 12 August 2023
bibliography: paper.bib

---

# Summary

In this work we introduce sparse-tensor contraction method in PyTorch analogous to `einsum` in NumPy.

# Statement of need

Among the many needs in high-performance scientific computing, two major problems arise: we must leverage sparse-data-structures and work with multidimensional-arrays (a parallelizable data-structure).
When working with multidimensional-arrays, a clear-and-concise and universal manipulation is the `einsum` function of NumPy, [@numpy:2020]. 
However, there is not much work in the intersection of these needs. I.e. those which manipulate sparse-tensors/arrays as `einsum` does. Therefore this work remedies this need.

# Overview of Functionality

As `einsum` stands for Einstein-Summation, `bisum` stands for Binary-Summation. The primary function of this package traces/contractions two tensors at a time (pair sequential-contraction is usually required for efficient contraction in multi-tensor traces) for types: sparse-sparse, sparse-dense, dense-sparse, and dense-dense (the original `einsum` function). This function intakes a string, list-of-tensors, xor tensor; to describe the partial-tracing procedure. Key features include:
1. Efficient Tensor Operations: `bisum` excels in performing a variety of tensor operations, including summation, contraction, and element-wise multiplication, on large dense data structures. While minimizing the memory usage.
2. Sparse Data Focus: the program capitalizes on the idea that many real-world data-sets contain numerous zero values. `bisum` optimizes computations by ignoring these zero values, significantly reducing the computational load and improving execution speed. By eliminating calculations involving zero values, `bisum` reduces memory usage and speeds up computation times, making it a valuable tool for applications involving massive data-sets. This work was originally motivated by [@Candanedo:2023].
3. Streamlined Syntax: `bisum` introduces a user-friendly syntax that simplifies the representation of tensor operations. This enables users to express complex mathematical operations concisely and intuitively, contributing to improved code readability and maintainability.
4. Application Flexibility `bisum` finds applications in a wide range of fields, including scientific research, engineering, machine learning, signal processing, and more. Its efficiency and ease of use make it a versatile choice for various computational tasks. This involves uses in Machine-Learning, Scientific-Simulations (e.g. physics, chemistry, engineering, and etc...), and Signal-Processing.
5. Optimized for Real Data: While many computations involve zero-padding, `bisum` focuses solely on real data values, eliminating the need to iterate over zero entries. This targeted approach ensures that the program's performance is optimized for sparse/dense, real-valued data-sets. Much real-world data can be made sparse with adequate transformations.
6. Compatibility: `bisum` can be easily integrated into existing code-bases and workflows, complementing other computational libraries and tools. It is integrated with the popular Machine-Learning library PyTorch, [@Paszke:2019].

`bisum` fills a crucial niche in the computational landscape by providing a specialized solution for efficient tensor operations on large sparse/dense/mixed data structures. Its focus on sparse data values that significantly boosts performance, making it an indispensable tool for tackling complex calculations in various domains. Whether it's accelerating machine learning tasks or enhancing scientific simulations, `bisum` offers a practical approach to optimizing computations while maintaining code simplicity and readability.

# Usage

## how to install

`bisum` is on the python-index, and therefore may be easily installed via the following command:
```console
pip install bisum
```

## import

`bisum` relies on sparse-tensors from PyTorch and therefore we import both libraries as such:
```python
import torch
from torch import einsum
from bisum import bisum
```
and on PyTorch we can determine where we would like the tensor to live (on CPU xor GPU)
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

\subsection*{`einsum`-like example}

We create natively dense random tensors (and can cast from to be sparse, via the `.to\_sparse()` command):
```python
A = torch.rand(8**3, device=device).reshape(8,8,8)
B = torch.rand(8**3, device=device).reshape(8,8,8)

torch.allclose( bisum("ijk,kjl", A.to_sparse(), B ), einsum("ijk,kjl", A, B ) )
torch.allclose( bisum("ijk,kjl", A.to_sparse(), B.to_sparse() ).to_dense(), einsum("ijk,kjl", A, B ) )
```

\subsection*{other label types example}

`bisum` traces 2 sparse-tensors (`torch.tensor` objects) via 3 Tracing-prescriptions:
1. `einsum`-string (like `numpy`, `str`, labelling each tensor axis)
2. `ncon` (used in the tensor-network community, list of 1d-`int`-`torch.tensor`, labelling each tensor axis, as described in [@Pfeifer:2014])
3. adjacency-matrix (as in `numpy.tensordot`, (2,n) 2d-`int`-`torch.tensor`, with n being the number of indices idenified between the two tensors)

Suppose we would like to compute the following partial-trace/tensor-contraction $C_{njwl} = A_{iksndj} B_{wklsdi}$:
```python
C_einsum = bisum("iksndj, wklsdi -> njwl", A, B)
C_ncon   = bisum([[-1,-2,-3,4,-5,6],[1,-2,3,-3,-5,-1]], A, B)
C_adjmat = bisum(torch.tensor([[0,1,2,4],[5,1,3,4]]), A, B)

print(torch.allclose(C_einsum, C_ncon) and torch.allclose(C_ncon, C_adjmat))
```
while the pure tensor-product, $\otimes$ is:
```python
import numpy as np

C_einsum = bisum("abcdef, ghijkl", A, B)
C_ncon   = bisum([], A, B)
C_adjmat = bisum(torch.tensor([]), A, B)

print(np.allclose(C_einsum, C_ncon) and np.allclose(C_ncon, C_adjmat))
```


## brief results when compared to `torch.einsum`

We did a quick comparison of this function (in the sparse-sparse mode) to PyTorch's native `einsum` function. The results of this comparison of relatively sparse-tensors is shown in fig. \ref{figurez}.

![This plot shows a timing comparison between the `torch.einsum` (solid line, averaged over 2 samples) function and the `bisum` (dots, averaged over 5 samples) function, for the sparse-sparse tensor contraction: $A_{qjwhkrjd}B_{krqljdmn}$ (each tensor of shape $\begin{pmatrix}14 & 14 & 14 & 14 & 14 & 14 & 14 & 14 \end{pmatrix}$) on a single CPU..](figure.png){ width=20% }

# Development Notes

Currently, `bisum` is in alpha-stage (0.2.0) with code on: [github](https://github.com/jcandane/bisum), and posted on the Python Package Index, or [PyPI](https://pypi.org/project/bisum/).
On here `bisum` has a MIT License.
Although, `bisum` is a useful extension of `einsum`-function, more improvements are desired.
Sparse-sparse matrix products on GPUs (tailor-made for dense-dense contractions) are relatively slow, sparse-dense contraction should be much faster.
Also functionality on block-sparse or jagged/ragged/PyTree (irregularity shaped) sparse-tensors is desired.

# Acknowledgements

This work is supported by Arizona State University Department of Physics and with help from Oliver Beckstein.

# References
