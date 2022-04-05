"""Script to run the best multi-class classifier"""
import matplotlib.pyplot as plt
import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from src.util.kernels import RBF, Polynomial, Intersection, Linear, Chi2
from sklearn.model_selection import train_test_split
import src.util.utils as ut
from src.methods.kernel_pca import *
from src.methods.oriented_edge_features import *


# Import the data
data_path = 'data/'
np.random.seed(42)

Xtr,Ytr,Xte = ut.read_data(data_path)
write_test_results = True
experiment_name = 'energy_hist_chi_1_5_f_1_f_2_using_8_directions'

if not write_test_results:

    Xtr, Xte, Ytr, Yte = train_test_split(Xtr, Ytr, train_size=0.8, shuffle=True)

# Transform into multilevel energy features
Xtr, Ytr = ut.augment_data(Xtr, Ytr)
filters = create_filters(8, 3, 1, lambda x,y : f1(x,y,1,3,1))
filters2 = create_filters(8, 3, 1, lambda x,y : f2(x,y,1,3,1))
both_filters = np.concatenate([filters, filters2])
mlef = multi_level_energy_features(8, both_filters, non_max=False)
Xtr = mlef.transform_all(Xtr) + 1e-6
Xte = mlef.transform_all(Xte) + 1e-6

kernel = Chi2(gamma=1.5).kernel

C = 1
classifier = MulticlassSVC(10, kernel, C, tol = 1e-2)
print('Training classifier C = ', C)
classifier.fit(Xtr, Ytr, verbose = True, use_weights=True, solver='cvxopt')

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