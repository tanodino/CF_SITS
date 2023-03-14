"""
Script to run our experiments using the kNN based CF generator from Karlsson et al 2020

Their CF models are implemented in a package called wildboar, that can be installed with pip.

For more info please see notes_on_Karlsons_code.md

author: tdrumond
date: 2023-03-13 16:39:09

"""
from wildboar.explain.counterfactual import KNeighborsCounterfactual as OriginalKNeighborsCounterfactual


class KNeighborsCounterfactual(OriginalKNeighborsCounterfactual):
    """override estimator validation to allow using our model"""
    def _validate_estimator(self, estimator, allow_3d=False):
        """skip validation"""
        pass


class ModelWrapper():
    """Wrap our models in order to provide the necessary interface for the nn counterfactual """

    def __init__(
            self, 
            model, 
            n_neighbors:int=1):
        self._model = model

        # n_neigbors is used in two situations:
        # 1) to compute n_clusters = n_samples// n_neighbors
        # 2) to compute the majority threshold: 1 + n_neighbors//n_classes
        self.n_neighbors = n_neighbors

        self.n_features_in_ = None
        self._fit_X = None
        self._y = None

    def fit(self, X, y):
        self._fit_X = X
        self._y = y
        self.n_features_in_ = X.shape[1]



