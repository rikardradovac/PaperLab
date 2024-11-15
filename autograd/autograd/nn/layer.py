import numpy as np
from ..core.tensor import Tensor
from .module import Module

class Layer(Module):
    def __init__(self, in_features, out_features, activation="tanh"):
        """
        Initialize a dense layer with specified activation.
        
        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            activation (str): Activation function - "tanh" or "sigmoid"
        """
        self.w = Tensor(np.random.normal(size=(in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.random.normal(size=(1, out_features)), requires_grad=True)
        self.activation = activation

    def __call__(self, x):
        """Forward pass through the layer."""
        model_output = x @ self.w + self.b

        if self.activation == "tanh":
            return model_output.tanh()
        elif self.activation == "sigmoid":
            return model_output.sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def parameters(self):
        """Return layer parameters."""
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, in_features, hidden_features, out_features):
        """
        Initialize a Multi-Layer Perceptron.
        
        Args:
            in_features (int): Number of input features
            hidden_features (int): Number of hidden layer features
            out_features (int): Number of output features
        """
        self.hidden = Layer(in_features, hidden_features)
        self.out = Layer(hidden_features, out_features, activation="sigmoid")

    def __call__(self, x):
        """Forward pass through the MLP."""
        return self.out(self.hidden(x))

    def parameters(self):
        """Return all MLP parameters."""
        return [p for layer in [self.hidden, self.out] for p in layer.parameters()]