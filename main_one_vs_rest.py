import numpy as np
import pandas as pd
from one_vs_rest import KernelSVC, MulticlassSVC
from kernels import RBF, Linear
from sklearn.model_selection import train_test_split
import utils as ut

# Import the data
data_path = 'data/'
nb_points = 1000 # For the first test use less points to limit computation time
Xtr = np.array(pd.read_csv(data_path + 'Xtr.csv',header=None,sep=',',usecols=range(3072)))[:nb_points]
Xte = np.array(pd.read_csv(data_path + 'Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv(data_path + 'Ytr.csv',sep=',',usecols=[1])).squeeze()[:nb_points]

# Undersample
# Xtr = ut.undersample(Xtr)

Ytr = 2*(Ytr == 4)-1
# Split the training dataset in train + validation
Xtr,Xval, Ytr, Yval = train_test_split(Xtr, Ytr, train_size=0.7, shuffle=True)
print(Xtr.shape)
# Create balanced training_set
Xtr_pos , Xtr_neg = Xtr[ Ytr == 1], Xtr[ Ytr == -1]
Ytr_pos, Ytr_neg = Ytr[Ytr == 1], Ytr[ Ytr == -1]
n = len(Xtr_neg) // len(Xtr_pos)
Xtr = np.concatenate([Xtr_pos]*n+ [Xtr_neg])
Ytr = np.concatenate([Ytr_pos]*n+ [Ytr_neg])
print(n, Xtr.shape)

# print('Preprocessing data')
# Xtr, transform = ut.images_to_pca(Xtr,n_components=3)
# Xval = transform(Xval)
# print(Xtr.shape, Ytr.shape)


# Train a binary classifier to start with
C = 1000
sigma = 1
class_weights = [ len(Ytr)/np.sum(Ytr == 1), len(Ytr)/np.sum(Ytr == -1)]
kernel = RBF(sigma=sigma).kernel
classifier = KernelSVC(C, kernel)

print('Training classifier (sigma, C) = ', sigma,C)
classifier.fit(Xtr, Ytr, verbose = True,  class_weights = None)

predictions, scores = classifier.predict(Xval), classifier.separating_function(Xval)

print((scores > 0).any())

print(ut.accuracy(Yval, predictions))




