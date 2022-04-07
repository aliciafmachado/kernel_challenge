"""
Grid search script example.
"""
from src.util.grid_search import GridSearch
import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from src.util.kernels import *
from src.methods.oriented_edge_features import *
import src.util.utils as ut
import pickle
import os


# Search for best parameters on gaussian kernel and then on polynomial kernel
def svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, 
                   parameters, tune="kernel", verbose=True, model_params=None, kernel_params=None):

    gs = GridSearch(parameters, model, kernel, tune=tune, model_params=model_params, kernel_params=kernel_params)
    gs.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    return gs


def __main__():
    # Load data
    data_path = 'data/'

    # You might want to change some parameters below
    name = 'example_tuning'
    augment = False
    seed = 42
    test_size = 0.2

    # Set seed
    np.random.seed(seed)

    # Using preprocessing in main_one_vs_all.py
    # Transform into oriented histograms
    print("Reading the data")
    Xtr, Ytr, _ = ut.read_data(data_path)
    X_train, X_val, y_train, y_val = ut.train_test_split(Xtr, Ytr, test_size=test_size, shuffle=True)

    print("Transforming the data")
    if augment:
        X_train, y_train = ut.augment_data(X_train, y_train)

    # Example using f1 filter only
    # But we could easily test with 1. f2; 2. f1 and f2;
    filters = create_filters(8, 3, 1, lambda x, y: f1(x, y, 1, 3, 1))
    mlef = multi_level_energy_features(8, filters, non_max=False)

    # Add epsilon to keep the values positive
    X_train = mlef.transform_all(X_train) + 1e-6
    X_val = mlef.transform_all(X_val) + 1e-6

    ### Normalize
    # Don't normalize when using GHI and Chi2 kernels !!
    # X_train, X_val = ut.normalize(X_train, X_val)

    # Search for best parameters using gaussian kernel
    kernel = 'chi2'
    print(f'Searching for best parameters using {kernel} kernel')
    
    # Case when searching kernels
    tune = "kernel"
    param = [1.0, 1.5, 2.0]
    name_param = 'gamma'
    parameters = {name_param: param}

    # If you want to search, e.g., C, you can do it like this:
    # parameters = {'C': [0.1, 1, 10, 100]}
    # And then, you have to set tune to 'model'
    # Other than that, you have to set default values for the kernel, e.g.:
    # kernel_params = {'gamma': 1}

    model = MulticlassSVC
    model_params={'C': 1, 'epsilon': 1e-3, 'tol': 1e-2}
    GS1 = svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, parameters, 
                        tune=tune, verbose=True, model_params=model_params, kernel_params=None)
    
    print(f'Best score: {GS1.best_score} using {GS1.best_hyperparameters}')

    # Save model and metrics
    with open(os.path.join(name + ".pickle"), 'wb') as f:
      pickle.dump(GS1, f)

if __name__ == '__main__':
    __main__()