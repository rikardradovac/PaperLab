import numpy as np
from ..core.tensor import Tensor

def test_Tensor_creation():
    x = Tensor(np.array([[2.0, 3.0]]))
    assert x.shape == (1, 2)
    assert not x.requires_grad

def test_Tensor_operations():
    x1 = Tensor(np.array([[2.0, 3.0]]))
    x2 = Tensor(np.array([[1.0, 4.0]]))
    w = Tensor(np.array([[-1.0], [1.2]]))

    # Test addition
    test_plus = x1 + x2
    assert np.allclose(test_plus.data, np.array([[3.0, 7.0]]))

    # Test subtraction
    test_minus = x1 - x2
    assert np.allclose(test_minus.data, np.array([[1.0, -1.0]]))

    # Test power
    test_power = x2**2
    assert np.allclose(test_power.data, np.array([[1.0, 16.0]]))

    # Test matrix multiplication
    test_matmul = x1 @ w
    assert np.allclose(test_matmul.data, np.array([[1.6]]))

def test_backward():
    x = Tensor(np.array([[2.0, 3.0]]))
    w = Tensor(np.array([[-1.0], [1.2]]), requires_grad=True)
    y = Tensor(np.array([[0.2]]))

    model_out = x @ w
    diff = model_out - y
    loss = diff**2

    loss.backward()

    assert np.allclose(w.grad, np.array([[5.6], [8.4]]))
    assert x.grad is None
    assert y.grad is None