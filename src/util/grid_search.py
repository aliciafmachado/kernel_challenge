"""
Grid search class.
"""
# TODO: test class

from src.util.utils import get_permutations, accuracy
import numpy as np


class GridSearch:
    def __init__(self, hyperparameters, methods):
        # TODO: maybe search for best preprocessing as well
        # self.preprocessing_pipelines = []
        self.methods = methods
        self.hyperparameters = hyperparameters
        self.best_hyperparameters = None
        self.best_predictor = None
        self.best_score = None
        self.all_predictors = []
        self.all_parameters = []
        self.all_methods = []
        self.all_scores = []

    def fit(self, X_train, y_train, X_test, y_test, verbose=True):
        for method in self.methods:
            for parameter in get_permutations(self.hyperparameters):
                m = method(**parameter)
                m.fit(X_train, y_train)
                self.all_predictors.append(m)
                self.all_scores.append(accuracy(m.predict(X_test), y_test))
                # Appending method and parameter which might not be ideal
                self.all_parameters.append((method, parameter))
                if verbose:
                    print(f'Using {method} with {parameter}')
                    print(self.all_scores[-1])

        best_score_idx = np.argmax(self.all_scores)

        self.best_score = self.all_scores[best_score_idx]
        self.best_predictor = self.all_predictors[best_score_idx]
        self.best_hyperparameters = self.all_parameters[best_score_idx]


    def predict(self, X_test):
        return self.best_predictor.predict(X_test)