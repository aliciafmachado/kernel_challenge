'''Classes of kernels'''

import numpy as np

class RBF :
    def __init__(self, sigma=1.) :
        self.sigma = sigma  ## the variance of the kernel

    def kernel(self, X, Y) :
        ## Input vectors X and Y of shape Nxd and Mxd
        return np.exp(-1/(2*self.sigma)*(-2*X@Y.T + np.sum(X**2, axis=1).reshape(-1,1) + np.sum(Y**2, axis=1).reshape(1,-1)))



class Linear :
    def __init__(self) :
        return

    def kernel(self, X, Y) :
        ## Input vectors X and Y of shape Nxd and Mxd
        N, d = X.shape
        return np.sum(Y * X.reshape(N, 1, d), axis=2)