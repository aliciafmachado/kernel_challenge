"""Kernel PCA."""

import numpy as np


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
        # Centralize
        K = K - np.mean(K, axis=0)

        eigen_vals, eigen_vecs = np.linalg.eigh(K)
        idx = eigen_vals.argsort()[::-1]

        # TODO: check for eigenvalues equal to 0
        self.eigen_vals = eigen_vals[idx][:self.n_components]
        self.eigen_vectors = eigen_vecs[:, idx][:, :self.n_components]

        return K @ np.divide(self.eigen_vectors, np.sqrt(self.eigen_vals))

    def transform(self, X):
        """
        Transform the data.
        """
        # Calculate kernel matrix
        K = self.kernel(X, self.X_fit)

        # Centralize
        K = K - np.mean(K, axis=0)

        return K @ np.divide(self.eigen_vectors, np.sqrt(self.eigen_vals))
