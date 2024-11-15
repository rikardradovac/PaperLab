import numpy as np

class Optimizer:
    def __init__(self, params):
        """
        Base optimizer class.
        
        Args:
            params: List of parameters to optimize
        """
        self.params = params

    def zero_grad(self):
        """Zero out parameter gradients."""
        for p in self.params:
            p.grad = np.zeros_like(p.data)

    def step(self):
        """Perform optimization step."""
        raise NotImplementedError("Unimplemented")

class SGD(Optimizer):
    def __init__(self, params, lr):
        """
        Stochastic Gradient Descent optimizer.
        
        Args:
            params: List of parameters to optimize
            lr (float): Learning rate
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        """Update parameters using SGD."""
        for p in self.params:
            if p.requires_grad:
                p.data -= self.lr * p.grad