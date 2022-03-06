import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA

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
