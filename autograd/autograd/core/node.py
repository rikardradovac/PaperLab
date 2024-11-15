
import numpy as np
from .functional import sigmoid

class Node:
    def __init__(self):
        pass

    def backward(self, grad_output):
        raise NotImplementedError("Unimplemented")

    def __repr__(self):
        return str(type(self))

class AdditionNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def backward(self, grad_output):
        self.left.grad = grad_output
        self.right.grad = grad_output
        self.right.backward(self.right.grad)
        self.left.backward(self.left.grad)

class SubtractionNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def backward(self, grad_output):
        self.left.grad = grad_output
        self.right.grad = -grad_output
        self.right.backward(self.right.grad)
        self.left.backward(self.left.grad)

class MatMulNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def backward(self, grad_output):
        self.left.grad = grad_output @ self.right.data.T
        self.right.grad = self.left.data.T @ grad_output
        self.right.backward(self.right.grad)
        self.left.backward(self.left.grad)

class PowNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def backward(self, grad_output):
        # result = left ** right
        # d_L/d_left = d_L/d_result * d_result/d_left = grad_output * right * left ** (right - 1)

        self.left.grad = (self.right * self.left.data ** (self.right - 1)) * grad_output
        self.left.backward(self.left.grad)


class TanhNode(Node):
    def __init__(self, left):
        self.left = left

    def backward(self, grad_output):
        # result = tanh(left)
        # d_L/d_left = d_L/d_result * d_result/d_left = grad_output * (1 - tanh(left) ** 2)

        self.left.grad = (1 - (np.tanh(self.left.data) ** 2)) * grad_output
        self.left.backward(self.left.grad)

class SigmoidNode(Node):
    def __init__(self, left):
        self.left = left

    def backward(self, grad_output):
        # result = sigmoid(left)
        # d_L/d_left = d_L/d_result * d_result/d_left = grad_output * (sigmoid(left) * (1 - sigmoid(left)))

        self.left.grad = grad_output * (sigmoid(self.left.data) * (1 - sigmoid(self.left.data)))
        self.left.backward(self.left.grad)

class BCELossNode(Node):
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        # forward for BCEloss
        self.output = -np.mean(self.y_true.data * np.log(self.y_pred.data) + (1 - self.y_true.data) * np.log(1 - self.y_pred.data))
        return self.output

    def backward(self, grad_output):
        # result = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        # d_L/d_y_pred = d_L/d_result * d_result/d_y_pred = grad_output * (-y_true / y_pred + (1 - y_true) / (1 - y_pred)) / num_elements

        num_elements = np.prod(self.y_pred.data.shape)
        grad_y_pred = grad_output * (-self.y_true.data / self.y_pred.data + (1 - self.y_true.data) / (1 - self.y_pred.data)) / num_elements
        self.y_pred.grad = grad_y_pred
        self.y_pred.backward(self.y_pred.grad)