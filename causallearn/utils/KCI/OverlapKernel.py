from __future__ import annotations


import numpy as np
from numpy import ndarray

from causallearn.utils.KCI.Kernel import Kernel



class OverlapKernel(Kernel):
    def __init__(self):
        Kernel.__init__(self)

    def kernel(self, X: ndarray, Y: ndarray | None = None):
        """
        Dirac kernel for discrete variables.
        
        Parameters:
        - X: array-like of shape (n_samples,)
        - Y: array-like of shape (n_samples,) or None

        Returns:
        - K: Kernel matrix of shape (n_samples, n_samples)
        """
        #.reshape(-1, 1)
        X = np.asarray(X).reshape(-1, 1)  # ensure column vector
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y)
        print(X.shape, Y.shape)
        result = (X == Y.T).astype(float)
    
        return result