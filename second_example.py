#PyTorch code for binary classification with a simple data set.
import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset
inputs = torch.randn(100, 10)  # 100 samples, each with 10 features
targets = torch.randint(0, 2, (100,))  # Binary classification task

# Define the neural network model
model = nn.Sequential(
    # Fully connected layer with 10 input features and 5 output features
    nn.Linear(10, 5), 
    # ReLU activation function 
    nn.ReLU(), 
    # Fully connected layer with 5 input features and 1 output 
    # feature (binary classification)        
    nn.Linear(5, 1)    
)

# Define the loss function and optimizer
# Binary Cross-Entropy loss function
criterion = nn.BCEWithLogitsLoss()
# Stochastic Gradient Descent (SGD) optimizer with learning rate 0.01  
optimizer = optim.SGD(model.parameters(), lr=0.01)  

# Training loop
# Train for 100 epochs
for epoch in range(100):  
    # Forward pass
    outputs = model(inputs)
    # Calculate the loss
    # Squeeze the outputs to match the shape of targets and ensure they're float
    loss = criterion(outputs.squeeze(), targets.float())  
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

print('Training complete.')
