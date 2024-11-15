import numpy as np
from .functional import sigmoid

class Tensor:
    def __init__(self, data, requires_grad=False, grad_fn=None):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        self.grad = None

    def __repr__(self):
        dstr = repr(self.data)
        if self.requires_grad:
            gstr = ", requires_grad=True"
        elif self.grad_fn is not None:
            gstr = f", grad_fn={self.grad_fn}"
        else:
            gstr = ""
        return f"Tensor({dstr}{gstr})"

    def item(self):
        return self.data.item()

    def __add__(self, right):
        from .node import AdditionNode
        new_data = self.data + right.data
        grad_fn = AdditionNode(self, right)
        if self.requires_grad or right.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def __sub__(self, right):
        from .node import SubtractionNode
        new_data = self.data - right.data
        grad_fn = SubtractionNode(self, right)
        if self.requires_grad or right.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def __matmul__(self, right):
        from .node import MatMulNode
        new_data = self.data @ right.data
        grad_fn = MatMulNode(self, right)
        if self.requires_grad or right.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def __pow__(self, right):
        from .node import PowNode
        if not isinstance(right, int) or right < 2:
            raise ValueError("Power must be an integer >= 2")
        grad_fn = PowNode(self, right)
        new_data = self.data**right
        if self.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def tanh(self):
        from .node import TanhNode
        new_data = np.tanh(self.data)
        grad_fn = TanhNode(self)
        if self.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def sigmoid(self):
        from .node import SigmoidNode
        new_data = sigmoid(self.data)
        grad_fn = SigmoidNode(self)
        if self.requires_grad:
            return Tensor(new_data, grad_fn=grad_fn, requires_grad=True)
        return Tensor(new_data, grad_fn=grad_fn)

    def backward(self, grad_output=None):
        if self.grad_fn is not None:
            if grad_output is None:
                self.grad_fn.backward(1)
            else:
                self.grad_fn.backward(self.grad)
        else:
            if self.requires_grad:
                self.grad = grad_output
            else:
                self.grad = None