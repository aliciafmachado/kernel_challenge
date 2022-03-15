"""Kernel PCA."""

import numpy as np
from scipy import linalg


class Kernel_PCA():
    def __init__(self, n_components, kernel_fn):
        """
        n_components: int, number of components to keep
        kernel_fn: function, kernel function
        """
        self.kernel = kernel_fn
        self.n_components = n_components

    def fit_and_transform(self, X):
        """
        Fit the model and transform the data.
        """
        self.X_fit = X
    
        # Compute gram matrix
        K = self.kernel(X, X)

        # Center the gram matrix
        aux = np.identity(K.shape[0]) - np.ones(K.shape) / K.shape[0]
        K_c = aux @ K @ aux

        eigenvals, eigenvecs = linalg.eigh(K_c)
        idxs = eigenvals.argsort()[::-1]

        self.eigenvals = eigenvals[idxs][:self.n_components]
        self.eigenvectors = eigenvecs[:, idxs][:, :self.n_components]

        non_zeros = np.flatnonzero(self.eigenvals)
        self.alphas = np.zeros(self.eigenvectors.shape)
        self.alphas[:, non_zeros] = np.divide(self.eigenvectors[:, non_zeros], np.sqrt(self.eigenvals[non_zeros]))

        return self.alphas

    def transform(self, X):
        """
        Transform the data.
        """
        # Calculate kernel matrix
        K = self.kernel(X, self.X_fit)

        # Center the kernel matrix
        aux0 = np.identity(K.shape[0]) - np.ones((K.shape[0], K.shape[0])) / K.shape[0]
        aux1 = np.identity(K.shape[1]) - np.ones((K.shape[1], K.shape[1])) / K.shape[1]
        K_c = aux0 @ K @ aux1

        return K_c @ self.alphas
