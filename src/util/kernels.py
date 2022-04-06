'''Classes of kernels'''

from random import betavariate
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


class GeneralizedHistogramIntersection :
    """
    Generalized Histogram Intersection in
    http://perso.lcpc.fr/tarel.jean-philippe/publis/jpt-icip05.pdf
    """

    def __init__(self, beta=1.0):
        # beta should be greater than 0
        self.beta = beta

    def kernel(self, X, Y):
        X_abs = np.abs(X) ** self.beta
        Y_abs = np.abs(Y) ** self.beta
        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[1]):
            kernel += np.minimum(X_abs[:,i].reshape(-1, 1), Y_abs[:,i].reshape(-1, 1).T)

        return kernel


class Chi2 :
    """
    Chi2 kernel.

    Attention: input should be strictly positive.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def kernel(self, X, Y):
        kernel = np.zeros((X.shape[0], Y.shape[0]))

        for i in range(X.shape[1]):
            x, y = X[:, i].reshape(-1, 1), Y[:, i].reshape(-1, 1)
            kernel += (x - y.T) ** 2 / (x + y.T)

        return np.exp(-self.gamma * kernel)


str_to_kernel = {
    "rbf": RBF,
    "linear": Linear,
    "polynomial": Polynomial,
    "intersection": Intersection,
    "ghi": GeneralizedHistogramIntersection,
    "chi2": Chi2,
}