import torch

# Create a tensor with requires_grad set to True
a = torch.randn(3, 3, requires_grad=True)

# Check if gradient tracking is enabled
print(a.requires_grad)

print(a)
