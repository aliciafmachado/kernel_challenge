""" Script to run and visualize Kernel PCA on the data"""

from matplotlib import pyplot as plt
from util.utils import read_data
from util.kernels import RBF, Linear
from methods.kernel_pca import Kernel_PCA
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import os


def main():
    save_path = "results/unsupervised_learning"
    Xtr_, Ytr, Xte_ = read_data()
    # kernel_fn = Linear().kernel
    kernel_fn = RBF(sigma=40).kernel

    # Normalize first
    mean = np.mean(Xtr_, axis=0)
    std = np.std(Xtr_, axis=0)
    Xtr_n = (Xtr_ - mean) / std

    # Apply the same to test
    Xte_n = (Xte_ - mean) / std

    # Apply PCA
    pca = Kernel_PCA(n_components=100, kernel_fn=kernel_fn)
    Xtr_tr = pca.fit_and_transform(Xtr_n)
    Xte_tr = pca.transform(Xte_n)

    # Save plot of eigenvalues and the first two dimensions after transformation
    plt.plot(pca.eigenvals)
    plt.savefig(os.path.join(save_path, 'kernel_pca_eigenvalues.png'))
    plt.clf()

    plt.scatter(Xtr_tr[:, 0], Xtr_tr[:, 1], c=Ytr)
    plt.savefig(os.path.join(save_path, 'pca_first_two_dimensions.png'))
    plt.clf()

if __name__ == "__main__":
    main()