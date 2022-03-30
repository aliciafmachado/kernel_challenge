'''Classes of kernels'''

import numpy as np


class RBF :
    def __init__(self, sigma=1.) :
        self.sigma = sigma  ## the variance of the kernel

    def kernel(self, X, Y) :
        ## Input vectors X and Y of shape Nxd and Mxd
        squared_norm = np.expand_dims(np.sum(X**2,axis=1),axis=1) + np.expand_dims(np.sum(Y**2,axis=1),axis=0)-2*np.einsum('ni,mi->nm',X,Y)
        return np.exp(-0.5*squared_norm/self.sigma**2)


class Linear :
    def __init__(self) :
        return

    def kernel(self, X, Y) :
        ## Input vectors X and Y of shape Nxd and Mxd
        return X @ Y.T


class Polynomial :
    def __init__(self, coef0=1, degree=2):
        self.coef0 = coef0
        self.degree = degree
    
    def kernel(self, X, Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return (X @ Y.T + self.coef0) ** self.degree


class Intersection :
    """
    Intersection kernel.
    """
    def __init__(self):
        return
    
    def kernel(self, X, Y):
        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[1]):
            kernel += np.minimum(X[:,i].reshape(-1, 1), Y[:,i].reshape(-1, 1).T)
            
        return kernel