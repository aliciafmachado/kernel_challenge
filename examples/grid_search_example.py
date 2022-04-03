"""
Grid search script example.
"""
from src.util.grid_search import GridSearch
import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from sklearn.model_selection import train_test_split
from src.util.kernels import *
from src.methods.oriented_edge_features import *
import src.util.utils as ut

# Best parameters: sigma: 10 (1, 10, 20, 50, 100) between the ones searched
#                  C: 1.0 (0.8, 0.9, 1.0, 1.1, 1.2) with fixed sigma and other params


# Search for best parameters on gaussian kernel and then on polynomial kernel
def svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, 
                   parameters, tune="kernel", verbose=True, model_params=None, kernel_params=None):

    gs = GridSearch(parameters, model, kernel, tune=tune, model_params=model_params, kernel_params=kernel_params)
    gs.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    return gs

def __main__():
    # Load data
    data_path = 'data/'

    seed = 42
    np.random.seed(seed)
    val_split = 0.2

    # Using preprocessing in main_one_vs_all.py
    # Transform into oriented histograms
    print("Reading the data")
    Xtr, Ytr, _ = ut.read_data(data_path)
    X_train, X_val, y_train, y_val = train_test_split(Xtr, Ytr, test_size=val_split, shuffle=True)

    print("Transforming the data")
    X_train, y_train = ut.augment_data(X_train, y_train)
    filters = create_filters(8, 3, 1, lambda x,y : f2(x,y,1,3,1))
    mlef = multi_level_energy_features(8, filters)

    # Add epsilon to keep the values positive
    X_train = mlef.transform_all(X_train) + 1e-6
    X_val = mlef.transform_all(X_val) + 1e-6

    ### Normalize
    # Don't normalize when using min and chi2 kernels
    # X_train, X_val = ut.normalize(X_train, X_val)

    # Search for best parameters using gaussian kernel
    kernel = 'chi2'
    print(f'Searching for best parameters using {kernel} kernel')
    
    gamma = [1.5]
    parameters= {'gamma': gamma}
    model = MulticlassSVC
    model_params={'C': 1, 'epsilon': 1e-3, 'tol': 1e-2}
    GS1 = svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, parameters, 
                        tune="kernel", verbose=True, model_params=model_params, kernel_params=None)
    print(f'Best score: {GS1.best_score} using {GS1.best_hyperparameters}')

    # # Search for best parameters using polynomial kernel
    # print("Searching for best parameters using polynomial kernel")
    # degree = [2, 3, 4, 5]
    # coef0 = [0.1, 0.5, 1, 2, 5]
    # parameters = {'degree': degree, 'coef0': coef0}
    # kernel = Polynomial
    # GS2 = svm_gridsearch(X_train, X_val, y_train, y_val,
    #                     model, kernel, parameters, tune="kernel", verbose=True)
    # print(f'Best score: {GS2.best_score} using {GS2.best_hyperparameters}')

    # Do preprocessing on Xte and save it
    # TODO: use predict fn from gridsearch or simply use the best parameters

if __name__ == '__main__':
    __main__()