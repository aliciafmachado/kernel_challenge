import numpy as np
import pandas as pd
from src.methods.one_vs_rest import KernelSVC, MulticlassSVC
from src.util.kernels import RBF, Linear
from sklearn.model_selection import train_test_split
import src.util.utils as ut


# Import the data
data_path = '../data/'

Xtr,Ytr,Xte = ut.read_data(data_path)


nb_points = 2000 # For the first test use less points to limit computation time
idx = np.random.choice(np.arange(len(Xtr)), size=nb_points)
Xtr, Ytr = Xtr[idx], Ytr[idx]

Xtr = ut.rgb_to_grayscale(Xtr)

Xtr,Xval, Ytr, Yval = train_test_split(Xtr, Ytr, train_size=0.7, shuffle=True)


# Undersample
# Xtr = ut.undersample(Xtr)


# print('Preprocessing data')
# Xtr, transform = ut.images_to_pca(Xtr,n_components=3)
# Xval = transform(Xval)
# print(Xtr.shape, Ytr.shape)


# Train a binary classifier to start with
C = 10000/len(Ytr)
sigma = 1
kernel = RBF(sigma=sigma).kernel
classifier = MulticlassSVC(10,kernel,C)
print('Training classifier (sigma, C) = ', sigma,C)
classifier.fit(Xtr, Ytr, verbose = True, use_weights=True)

predictions, scores = classifier.predict(Xval)

print((scores > 0).any())

print(ut.accuracy(Yval, predictions))




