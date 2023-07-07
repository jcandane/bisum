import torch

A = torch.rand((5,8,4,4,6,6))

l = torch.tensor([0,1,2,4])

## slice
#A = A[]
#print( torch.sum(A, (1,2)) )

# Input tensor
tensor = torch.randn(2, 3, 4, 4, 4, 4)  # Example input tensor

# Extract diagonal elements from specific dimensions
reshaped_tensor = tensor.view(tensor.size()[:3] + (-1,))
diagonal_tensor = torch.diagonal(reshaped_tensor, dim1=-2, dim2=-1)

print(diagonal_tensor.shape)  # Output: (2, 3, 4)