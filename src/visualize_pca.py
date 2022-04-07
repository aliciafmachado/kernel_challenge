""" Script to run and visualize Kernel PCA on the data"""

from matplotlib import pyplot as plt
from util.utils import read_data
from util.kernels import RBF
from methods.kernel_pca import Kernel_PCA
import numpy as np
import os


def main():
    save_path = "results/unsupervised_learning/"
    data_path = 'data/'
    Xtr_, Ytr, _ = read_data(data_path)
    sigma = 10
    kernel_fn = RBF(sigma=sigma).kernel

    # Normalize first (when not using chi2 or GHI)
    mean = np.mean(Xtr_, axis=0)
    std = np.std(Xtr_, axis=0)
    Xtr_n = (Xtr_ - mean) / std

    # Apply PCA
    pca = Kernel_PCA(n_components=100, kernel_fn=kernel_fn)
    Xtr_tr = pca.fit_and_transform(Xtr_n)

    # Save plot of eigenvalues and the first two dimensions after transformation
    plt.plot(pca.eigenvals)
    plt.savefig(os.path.join(save_path, f'kernel_pca_eigenvalues_sigma{sigma}.png'))
    plt.clf()

    plt.scatter(Xtr_tr[:, 0], Xtr_tr[:, 1], c=Ytr)
    plt.savefig(os.path.join(save_path, f'pca_first_two_dimensions_sigma{sigma}.png'))
    plt.clf()

if __name__ == "__main__":
    main()