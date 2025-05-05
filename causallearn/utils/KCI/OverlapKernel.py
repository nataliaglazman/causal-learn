from __future__ import annotations


import numpy as np
from numpy import ndarray

from causallearn.utils.KCI.Kernel import Kernel
from scipy.spatial.distance import pdist, squareform



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
        print('using overlap kernel')
        X = np.asarray(X)
        if Y is None:
            Y = X
        else:
            Y = np.asarray(Y)
        # result = (X == Y.T).astype(float)
        result = squareform(pdist(X, 'jaccard'))
    
        return result