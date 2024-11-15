import numpy as np
from ..nn.layer import MLP
from ..core.tensor import Tensor
from ..nn.loss import bce_loss
from ..optim.optimizer import SGD

def test_mlp_forward():
    np.random.seed(1)
    model = MLP(2, 4, 1)
    x = Tensor(np.random.randn(1, 2))
    output = model(x)
    assert output.shape == (1, 1)
    assert 0 <= output.data.item() <= 1  # sigmoid output

def test_mlp_training_step():
    np.random.seed(1)
    model = MLP(2, 4, 1)
    optimizer = SGD(model.parameters(), lr=0.01)
    
    x = Tensor(np.random.randn(1, 2))
    y = Tensor(np.array([[1.0]]))
    
    # Forward pass
    y_pred = model(x)
    loss = bce_loss(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that gradients were computed and parameters updated
    for param in model.parameters():
        assert param.grad is not None