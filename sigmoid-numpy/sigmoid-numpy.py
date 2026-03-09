import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x).astype(float)
    a = (1 / (1 + np.exp(-x))).tolist()
    return a
    
    