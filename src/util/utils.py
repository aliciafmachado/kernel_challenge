"Util functions."

import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import pandas as pd
import os
from scipy.fft import fft2, fftshift



def accuracy(y, pred):
    return np.sum(y == pred)/len(y)

def image_to_hist(im, bins,low,up):
    hR = np.histogram(im[:1024], bins = bins, range = (low, up), density = True)
    hG = np.histogram(im[1024:2048], bins=bins, range= (low, up), density = True)
    hB = np.histogram(im[2048:], bins=bins, range= (low, up), density = True)

    return hR,hG,hB

def images_to_hist(images, bins, low, up):
    histograms = []
    for im in images:
        hr, hg, hb = image_to_hist(im, bins, low, up)
        concat_hist = np.concatenate([hr[0],hg[0],hb[0]])
        histograms.append(concat_hist)
    return np.array(histograms)

def images_to_pca(train, n_components):
    pca = PCA(n_components)
    pca.fit(X=train)
    print(pca.explained_variance_ratio_.sum())
    return pca.transform(train), pca.transform

def undersample(train):
    return train[:,::2]

def read_data(data_path="data/"):
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
    dataframe.to_csv(results_path+results_name, index_label='Id')

    print("Results ready for submission!")


def rgb_to_grayscale(Xtr):
    """
    averages over the three channels to reduce dimension and have a grayscale image
    :param Xtr:
    :return: Xtr_gray
    """

    Xtr_gray = np.mean(Xtr.reshape(len(Xtr),3,-1),1)
    return Xtr_gray

def normalize(Xtr,Xte):
    mean = np.mean(Xtr, axis=0)
    std = np.std(Xtr, axis=0)
    Xtr_n = (Xtr - mean) / std

    # Apply the same to test
    Xte_n = (Xte - mean) / std
    return Xtr_n,Xte_n

def image_to_fourier(img):
    """

    :param img: 3072 array
    :return:
    """

    fourierR = np.abs(fftshift(fft2(img[:1024].reshape(32,32))))
    fourierG = np.abs(fftshift(fft2(img[1024:2048].reshape(32,32))))
    fourierB = np.abs(fftshift(fft2(img[2048:].reshape(32,32))))

    return np.concatenate([fourierR.flatten(), fourierG.flatten(),fourierB.flatten()], axis=0)

def transform_to_fourier(Xtr):

    fouriers = []
    for img in Xtr:
        fouriers.append(image_to_fourier(img))
    return np.array(fouriers)