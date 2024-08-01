import os

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from tabularbench.config.config_run import ConfigRun
from tabularbench.models.carte.src.carte_estimator import CARTEClassifier
from tabularbench.models.carte.src.carte_table_to_graph import Table2GraphTransformer
from tabularbench.results.prediction_metrics import PredictionMetrics


class Trainer(BaseEstimator):

    def __init__(
            self, 
            cfg: ConfigRun,
            model: torch.nn.Module,
            n_classes: int,
            feature_names: list[str]
        ) -> None:

        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)

        fixed_params = dict()
        fixed_params["num_model"] = 10 # 10 models for the bagging strategy
        fixed_params["disable_pbar"] = False # True if you want cleanness
        fixed_params["random_state"] = 0
        fixed_params["device"] = "cuda"
        fixed_params["n_jobs"] = 10

        self.cfg = cfg
        self.n_classes = n_classes
        self.feature_names = feature_names

        self.preprocessor = Table2GraphTransformer()
        self.estimator = CARTEClassifier(**fixed_params) # CARTERegressor for Regression



    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):

        X_df = pd.DataFrame(x_train, columns=self.feature_names)
        X_train = self.preprocessor.fit_transform(X_df, y=y_train)
        self.estimator.fit(X=X_train, y=y_train)

    
    def test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> PredictionMetrics:

        X_df = pd.DataFrame(x_test, columns=self.feature_names)
        X_test = self.preprocessor.transform(X_df)
        y_pred = self.estimator.predict_proba(X_test)
        
        prediction_metrics = PredictionMetrics.from_prediction(y_pred, y_test, self.cfg.task)
        return prediction_metrics