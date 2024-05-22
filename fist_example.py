#Generation of a tensor and activation of gradient tracking
import torch

# Define gradient enabled tensors
x = torch.tensor([0.0, 2.0, 8.0], requires_grad=True)

y = torch.tensor ([5.0, 1.0, 7.0], requires_grad=True)

# Perform multiplication operation
z = x * y

# Calculate gradients with external gradient
z.backward(torch.FloatTensor ([1.0, 1.0, 1.0]))

# Access calculated gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)

print("Tensor z:", z)
