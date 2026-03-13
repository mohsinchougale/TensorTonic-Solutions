import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here

    def bce_loss(y, p):
        loss = (y * np.log(p)) + ((1 - y)*(np.log(1-p)))
        return -np.mean(loss)

    
    def calculate_gradients(X,y,p):
        dL_dw = np.mean(X.T @ (p-y))
        # print("dL_dw = ", dL_dw)

        dL_db = np.mean(p-y)
        # print("dL_db = ", dL_db)

        return dL_dw, dL_db

        
    def param_update(w, b, lr, dL_dw, dL_db):
        w_updated = w - (lr*dL_dw)
        b_updated = b - (lr*dL_db)

        return w_updated, b_updated
        
    
    
    # rng = np.random.default_rng(seed=42)
    

    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    
    print(X, X.shape)
    print(y, y.shape)
    print(w, w.shape)
    # print(b, b.shape)


    for _ in range(steps):
        # print(f"Step {_ + 1}")
        # print("Output = ")
        linear_output = X@w + b
        # print(linear_output, linear_output.shape)
        p = _sigmoid(linear_output) #sigmoid_activation
        # print("Sigmoid Output = ", p, p.shape)
        
    
        L = bce_loss(y,p)
        
        # print("Binary Cross Entropy Loss = ", L)
    
        dL_dw, dL_db = calculate_gradients(X,y,p)
    
        
    
        w, b = param_update(w, b, lr, dL_dw, dL_db)
    
    print("Updated params: ")
    print("w = ", w)
    print("b = ", b)

    return (w,b)
    