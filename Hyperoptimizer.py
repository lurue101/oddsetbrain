import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from model import OddsetNet
from sklearn.model_selection import train_test_split

from hyperopt import tpe, STATUS_OK,Trials,hp,fmin
import numpy as np
import torch as T

class Hyperoptimizer():
    def __init__(self, model:nn.Module, X: T.Tensor, y: T.Tensor, val_size: 0.2):
        self.model = model
        self.val_size = 0.2



    def split_and_normalize_train_set(self,X,y,k=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size)
        normalizer = StandardScaler().fit(X_train)
        X_train = normalizer.transform(X_train)
        X_val = normalizer.transform(X_val)


    def objective(self,
            params,
    ):
        """Objective function for OddsetNet Hyperparameter Tuning with TPE algorithm"""

        model = OddsetNet(**params)
        scores_dict, _ = self.model.train_model(50,
                                           trainloader,
                                           X_val,
                                           y_val,
                                           y_odds_val,
                                           "adam",
                                           nn.CrossEntropyLoss(),
                                           Accuracy(task="multiclass", num_classes=3))

        # Extract the best score
        best_acc = scores_dict["best_acc"]

        # Loss must be minimized
        loss = 1 - best_acc

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}

    def objective_cv(self,
            params,
    ):
        """Objective function for cross-validated OddsetNet Hyperparameter Tuning"""

        model = OddsetNet(**params)
        scores_dict = model.cross_validation(50, np.concatenate([X_train, X_val]),
                                             np.concatenate([y_train, y_val]),
                                             np.concatenate([y_odds_train, y_odds_val]),
                                             batch_size=4,
                                             k=5)

        # Extract the best score
        best_mean_acc = scores_dict["mean_best_acc"]

        # Loss must be minimized
        loss = 1 - best_mean_acc

        # Dictionary with information for evaluation
        return {'loss': loss, 'params': params, 'status': STATUS_OK}