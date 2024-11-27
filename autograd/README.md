# Autograd

A lightweight automatic differentiation library implemented from scratch in Python. This library provides basic neural network building blocks with automatic gradient computation.

## Features

- Tensor operations with automatic gradient tracking
- Basic neural network layers and MLP implementation
- Common activation functions (tanh, sigmoid)
- Binary Cross Entropy loss
- SGD optimizer
- Automatic backpropagation

## Example Usage

```python
from autograd.core.tensor import Tensor
from autograd.nn.layer import MLP
from autograd.optim.optimizer import SGD
from autograd.nn.loss import bce_loss

# Create a simple MLP model
model = MLP(in_features=2, hidden_features=4, out_features=1)
optimizer = SGD(model.parameters(), lr=0.01)

# Sample data
x = Tensor([[0.5, 0.2]])
y = Tensor([[1.0]])

# Forward pass
y_pred = model(x)
loss = bce_loss(y_pred, y)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()
```




## Components

- **Core**: Tensor operations and automatic differentiation
- **NN**: Neural network layers and loss functions
- **Optim**: Optimization algorithms
- **Tests**: Unit tests for components

## Requirements

- NumPy

## License

MIT