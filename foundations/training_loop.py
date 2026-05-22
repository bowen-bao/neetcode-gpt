import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n, n_features = X.shape 
        w = np.zeros(n_features)
        b = 0.0 

        for _ in range(epochs):
            # forward pass 
            y_hat = X @ w + b

            # gradients 
            diff = y_hat - y 
            dw = (2/n) * X.T @ diff 
            db = (2/n) * np.sum(diff)

            # update 
            w = w - lr * dw
            b = b - lr * db 
        
        return (np.round(w, 5), round(b, 5))