"""
Grid search script example.
"""
from src.util.grid_search import GridSearch
import numpy as np
from src.methods.one_vs_rest import MulticlassSVC
from sklearn.model_selection import train_test_split
from src.util.kernels import RBF, Polynomial
from src.methods.oriented_edge_detection import Xtr_to_energy_hist, create_filters
import src.util.utils as ut


# Search for best parameters on gaussian kernel and then on polynomial kernel
def svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, 
                   parameters, tune="kernel", verbose=True):

    gs = GridSearch(parameters, model, kernel, tune=tune, model_params={'C': 1, 'epsilon': 1e-3, 'tol': 1e-2})
    gs.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    return gs

def __main__():
    # TODO: There is a bug here even though i followed the same structure on main_one_vs_rest.py :/
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
    filters = create_filters(8,1,0.5,1,5,1)
    X_train = Xtr_to_energy_hist(X_train,filters,15,0.5)
    X_val = Xtr_to_energy_hist(X_val,filters, 15,0.5)

    ### Normalize
    X_train, X_val = ut.normalize(X_train, X_val)

    # Search for best parameters using gaussian kernel
    print("Searching for best parameters using gaussian kernel")
    # sigmas = [1, 10, 20, 50, 100]
    sigmas = [10, 20]
    parameters= {'sigma': sigmas}
    kernel = RBF
    model = MulticlassSVC
    GS1 = svm_gridsearch(X_train, X_val, y_train, y_val, model, kernel, parameters, 
                        tune="kernel", verbose=True)
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