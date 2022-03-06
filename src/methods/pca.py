"""Normal PCA without kernels."""

import numpy as np


class PCA():
    def __init__(self, n_components):
        """
        n_components: int, number of components to keep
        """
        self.n_components = n_components
        self.eigenvectors_subset = None
        self.vars = None

    def fit_and_transform(self, X):
        """
        Fit the model and transform the data.

        The data is supposed to be centered in 0
        """
        cov_m = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_m)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.eigenvectors_subset = eigenvectors[:, :self.n_components]
        self.vars = eigenvalues[:self.n_components]

        return X @ self.eigenvectors_subset
        

    def transform(self, X):
        """
        Transform the data.
        """
        return X @ self.eigenvectors_subset
