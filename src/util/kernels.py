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

# class MultivariateKernel :
#     def __init__(self, kernel, args):
#         self.kernel = kernel(**args).kernel
    
#     def kernel(self, X, Y) :
#         ## Input vectors X and Y of shape Nxdxd and Mxdxd
#         d = X.shape[-1]
#         # Take all x concatenated (certain ordering)
#         x_x =
#         y_x =

#         # Take all y concatenated (other ordering)
#         x_y = ... 

#         # TODO: Sum or product of the kernels?
#         # For gaussian i think it's product so i will keep that way
#         # Pointwise multiplication:
#         return self.kernel(x_x, y_x) * self.kernel(x_y, y_x)