#PyTorch Training Loop with Custom Neural Network and Loss Function
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Example dataset
inputs = torch.randn(1000, 784)  # Example: 1000 samples, each with 784 features
targets = torch.randint(0, 10, (1000,))  # Example: 1000 labels for 10 classes

# Create DataLoader
dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the neural network model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define your model architecture here
        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

# Create an instance of the custom model
model = CustomModel()

# Define the custom loss function
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        # Implement custom loss function logic here
        # Example: Cross-Entropy Loss
        loss = nn.CrossEntropyLoss()(outputs, targets)  
        return loss

# Create an instance of the custom loss
custom_loss = CustomLoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Define the number of epochs
num_epochs = 10  # For example, train for 10 epochs

# Training loop
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = custom_loss(outputs, targets)

        # Backward pass (AutoDiff)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

print('Training complete.')
