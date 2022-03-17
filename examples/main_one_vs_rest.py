"""Script to run a multi-class classifier testing different pre_processing techniques"""

import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from src.util.kernels import RBF
from sklearn.model_selection import train_test_split
import src.util.utils as ut

# Import the data
data_path = '../data/'

Xtr,Ytr,Xte = ut.read_data(data_path)
nb_points = 5000  # Work on a subpart of the data to limit computation time. MAX = 5000
idx = np.random.choice(np.arange(len(Xtr)), size=nb_points)
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



### Apply PCA
# kernel_fn = RBF(sigma=10).kernel
# pca = Kernel_PCA(n_components=50, kernel_fn=kernel_fn)
# Xtr = pca.fit_and_transform(Xtr)
# Xte = pca.transform(Xte)

### Transform into orientation histograms
from src.methods.oriented_edge_detection import Xtr_to_energy_hist, create_filters
filters = create_filters(8,1,0.5,1,5,1)
Xtr = Xtr_to_energy_hist(Xtr,filters,15,0.5)
Xte = Xtr_to_energy_hist(Xte,filters, 15,0.5)

### Normalize
Xtr, Xte = ut.normalize(Xtr,Xte)




#---------------------------------------------------------------------------------------
# -------------- TRAINING --------------------------------------------------------------
# -------------------------------------------------------------------------------------


# Train a binary classifier to start with
C = 1
sigma = 10
kernel = RBF(sigma=sigma).kernel
classifier = MulticlassSVC(10,kernel,C, tol = 1e-2)
print('Training classifier (sigma, C) = ', sigma,C)
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
    print(ut.accuracy(Yte, predictions))

else :
    # Write the predictions on Xte
    predictions, scores = classifier.predict(Xte)
    print((scores > 0).any())
    ut.save_results(predictions,results_name='Yte_pred_'+ experiment_name+'.csv' , results_path='../data/')



