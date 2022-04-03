"""Script to run a multi-class classifier testing different pre_processing techniques"""
import matplotlib.pyplot as plt
import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from src.util.kernels import RBF, Polynomial, Intersection, Linear, Chi2
from sklearn.model_selection import train_test_split
import src.util.utils as ut
from src.methods.kernel_pca import *
from src.methods.oriented_edge_features import *


# Import the data
data_path = '../data/'
np.random.seed(42)

Xtr,Ytr,Xte = ut.read_data(data_path)
nb_points = 5000  # Work on a subpart of the data to limit computation time. MAX = 5000
idx = np.random.choice(np.arange(len(Xtr)), size=nb_points, replace=False)
Xtr, Ytr = Xtr[idx], Ytr[idx]


write_test_results = False
experiment_name = 'energy_hist_sigma_10_C_1_norm'

if not write_test_results:

    Xtr, Xte, Ytr, Yte = train_test_split(Xtr, Ytr, train_size=0.8, shuffle=True)

#--------------------------------------------------------------------------------
#--------- PREPROCESSING --------------------------------------------------------
#--------------------------------------------------------------------------------

# Uncomment the part you want to experiment with

### Transform images with sift and bovw
# Xtr = ut.rgb_to_grayscale(Xtr)
# Xtr = transform_bovw_vectors(Xtr)

### Transform into fourier representation of
# Xtr = ut.transform_to_fourier(Xtr)
# Xte = ut.transform_to_fourier(Xte)


### Transform into histogram representation
# Xtr = ut.images_to_hist(Xtr, 30, -0.1,0.1)
# Xte = ut.images_to_hist(Xte, 30, -0.1, 0.1)


# ### Transform into orientation histograms
# filters = create_filters(8,1,3,1,3,1)
# en_hist = energy_hist(filters,nbins=5,bound=0.5)
# Xtr = en_hist.transform_all(Xtr) + 1e-5
# Xte = en_hist.transform_all(Xte) + 1e-5

# Transform into multilevel energy features
filters = create_filters(8, 3, 1, lambda x,y : f2(x,y,1,3,1))
mlef = multi_level_energy_features(8 ,filters, 15, 0.5)
Xtr = mlef.transform_all(Xtr) +1e-6
Xte = mlef.transform_all(Xte) + 1e-6

### Apply PCA
# kernel_fn = Intersection().kernel
# pca = Kernel_PCA(n_components=50, kernel_fn=kernel_fn)
# Xtr = pca.fit_and_transform(Xtr)
# Xte = pca.transform(Xte)
#
# plt.scatter(Xtr[:,0], Xtr[:,1], c=Ytr)
# plt.savefig('kpca.png')
# 1/0
### Sanity check for the shape
print(f'Shape of features {np.shape(Xtr)}')

### Normalize
# Xtr, Xte = ut.normalize(Xtr,Xte)




#---------------------------------------------------------------------------------------
# -------------- TRAINING --------------------------------------------------------------
# -------------------------------------------------------------------------------------


# Train a binary classifier to start with

# Choose a kernel

# sigma = 10
# kernel = RBF(sigma=sigma).kernel

# kernel = Intersection().kernel # Don"t normalize if you use this!
# kernel = Linear().kernel
kernel = Chi2().kernel

C = 1
classifier = MulticlassSVC(10,kernel,C, tol = 1e-2)
print('Training classifier C = ', C)
classifier.fit(Xtr, Ytr, verbose = True, use_weights=True, solver='cvxopt')

#------------------------------------------------------------------------------
#--------------- TESTING ------------------------------------------------------
#------------------------------------------------------------------------------

# Compute the accuracy on the training data
predictions, scores = classifier.predict(Xtr)
print((scores > 0).any())
print("Training accuracy ;", ut.accuracy(Ytr, predictions))

# If we didn't train on all training data
if not write_test_results :
    predictions, scores = classifier.predict(Xte)
    print((scores > 0).any())
    print("Validation accuracy ;",ut.accuracy(Yte, predictions))
    print('Confusion matrix')
    print(ut.compute_confusion_matrix(Yte,predictions,10))

else :
    # Write the predictions on Xte
    predictions, scores = classifier.predict(Xte)
    print((scores > 0).any())
    ut.save_results(predictions,results_name='Yte_pred_'+ experiment_name+'.csv' , results_path='../data/')



