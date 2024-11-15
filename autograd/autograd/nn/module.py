import numpy as np

class Module:
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        """Return list of trainable parameters."""
        raise NotImplementedError("Method for getting parameters is not implemented")

    def disable_grad(self):
        """Disable gradient computation for all parameters."""
        for p in self.parameters():
            p.requires_grad = False