import numpy as np
from ..core.tensor import Tensor
from ..core.node import Node

class BCELossNode(Node):
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        """Compute Binary Cross Entropy Loss."""
        self.output = -np.mean(
            self.y_true.data * np.log(self.y_pred.data) + 
            (1 - self.y_true.data) * np.log(1 - self.y_pred.data)
        )
        return self.output

    def backward(self, grad_output):
        """Compute gradients for BCE Loss."""
        num_elements = np.prod(self.y_pred.data.shape)
        grad_y_pred = grad_output * (
            -self.y_true.data / self.y_pred.data + 
            (1 - self.y_true.data) / (1 - self.y_pred.data)
        ) / num_elements
        self.y_pred.grad = grad_y_pred
        self.y_pred.backward(self.y_pred.grad)

def bce_loss(y_pred, y_true):
    """Binary Cross Entropy Loss function."""
    return Tensor(
        BCELossNode(y_pred, y_true).forward(), 
        grad_fn=BCELossNode(y_pred, y_true)
    )