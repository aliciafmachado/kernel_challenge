"Util functions."

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd
import os


def accuracy(y, pred):
    return np.sum(y == pred)/len(y)

def images_to_hist(images, bins,low,up):
    histograms = []
    for x in tqdm(images):
        histograms.append(np.histogram(images, bins = bins, range = (low, up), density = True))
    return np.array([histograms])

def images_to_pca(train, n_components):
    pca = PCA(n_components)
    pca.fit(X=train)
    print(pca.explained_variance_ratio_.sum())
    return pca.transform(train), pca.transform

def undersample(train):
    return train[:,::2]

def read_data(data_path="data"):
    """
    Read the data from the data_path.
    """
    Xtr = np.array(pd.read_csv(os.path.join(data_path, 'Xtr.csv'),header=None,sep=',',usecols=range(3072)))
    Xte = np.array(pd.read_csv(os.path.join(data_path, 'Xte.csv'),header=None,sep=',',usecols=range(3072)))
    Ytr = np.array(pd.read_csv(os.path.join(data_path, 'Ytr.csv'),sep=',',usecols=[1])).squeeze()

    return Xtr, Ytr, Xte


def save_results(Yte, results_name="Yte_pred.csv", results_path="data"):
    """
    Read the data from the data_path.
    """

    Yte = {'Prediction' : Yte}
    dataframe = pd.DataFrame(Yte)
    dataframe.index += 1
    dataframe.to_csv(results_name, index_label='Id')

    print("Results ready for submission!")