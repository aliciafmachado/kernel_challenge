"""
Grid search class.
"""
# TODO: test class

from src.util.utils import get_permutations, accuracy
import numpy as np


class GridSearch:
    def __init__(self, hyperparameters, model, kernel, tune="kernel", 
                model_params=None, kernel_params=None):
        # TODO: maybe search for best preprocessing as well
        # self.preprocessing_pipelines = []
        self.model = model
        self.kernel = kernel
        self.model_params = model_params
        self.kernel_params = kernel_params
        self.tune = tune
        self.hyperparameters = hyperparameters
        self.best_hyperparameters = None
        self.best_predictor = None
        self.best_score = None
        self.all_predictors = []
        self.all_parameters = []
        self.all_methods = []
        self.all_scores = []

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):

        for parameter in get_permutations(self.hyperparameters):
            if self.tune == "kernel":
                kernel_params = parameter
                model_params = self.model_params
            elif self.tune == "model":
                model_params = parameter
                kernel_params = self.kernel_params

            kernel = self.kernel(**kernel_params).kernel
            m = self.model(10, kernel, **model_params)
            m.fit(X_train, y_train, verbose=True, use_weights=True, solver='cvxopt')
                
            self.all_predictors.append(m)
            predictions, _ = m.predict(X_test)
            self.all_scores.append(accuracy(y_test, predictions))
            # Appending method and parameter which might not be ideal
            self.all_parameters.append(parameter)

            if verbose:
                print(f'Using {self.model} with {parameter}')
                print(self.all_scores[-1])

        best_score_idx = np.argmax(self.all_scores)

        self.best_score = self.all_scores[best_score_idx]
        self.best_predictor = self.all_predictors[best_score_idx]
        self.best_hyperparameters = self.all_parameters[best_score_idx]


    def predict(self, X_test):
        return self.best_predictor.predict(X_test)