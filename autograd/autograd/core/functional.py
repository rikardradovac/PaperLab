import numpy as np

def sigmoid(x):
    """Helper function to compute the sigmoid."""
    return 1 / (1 + np.exp(-x))