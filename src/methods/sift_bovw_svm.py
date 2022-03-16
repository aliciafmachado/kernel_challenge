""" Use SIFT descriptors and bag of visual words to compute features vectors"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import src.util.utils as ut



def compute_sift(Xtr_im):
    """from an image, return the sift descriptors"""

    img = Xtr_im.reshape(32, 32)

    img = (img - np.min(img)) * 255 / np.max(img)
    img = img.astype(np.uint8)


    sift = cv2.SIFT_create()
    keypoint, descriptor = sift.detectAndCompute(img, None)
    return descriptor


def compute_all_descriptors(Xtr):
    """

    :param Xtr: array of gray images (N*1024)
    :return: the descriptors for each instance and the memory of how many descriptor
    there is for each image (useful to keep track of wich descriptor belongs to who)
    """

    descriptors = []
    memory = []
    for img in Xtr:
        d = compute_sift(img)
        if d is not None :
            descriptors.append(d)
            memory.append(len(d))
        else :
            memory.append(0)
    return np.concatenate(descriptors), memory



from sklearn.cluster import KMeans

def compute_clusters(features,nb_centroids):
    """

    :param features: array of features
    :param nb_centroids: nb of clusters to do
    :return: the kmeans instance and cluster-labels for each instance
    """

    kmeans = KMeans(nb_centroids).fit(features)
    labels = np.argmin(kmeans.transform(features), axis=1)
    return kmeans, labels


def compute_bovw(labels, memory, nb_centroids):
    """

    :param labels: each cluster label for each descriptor instances
    :param memory: the array of how many descriptors belong to the i-th image
    :param nb_centroids: the number of cluster that was used
    :return: the bovw representation for each original image instance
    """

    feature_vectors = []
    cpt = 0
    for l in memory:
        if l==0:
            feature_vectors.append(np.zeros(nb_centroids))
        else:
            get_kp = labels[cpt:cpt+l]
            hist = np.histogram(get_kp,bins = np.arange((nb_centroids+1)))
            feature_vectors.append(hist[0])
            cpt+=l
    return np.array(feature_vectors)



# Then normalize_features vector and try SVM classification on it

def transform_bovw_vectors(Xtr):
    """

    :param Xtr: the set of images
    :return: the bovw features
    """
    all_descriptors, memory = compute_all_descriptors(Xtr)
    kmeans, labels = compute_clusters(all_descriptors, 30)
    features_vectors = compute_bovw(labels, memory, 30)

    return features_vectors

def main():
    Xtr, Ytr, Xte = ut.read_data('../../data/')
    Xtr = ut.rgb_to_grayscale(Xtr)
    features_vectors = transform_bovw_vectors(Xtr)

if __name__ == "__main__":
    main()
