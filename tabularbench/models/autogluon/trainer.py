import os

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator

from tabularbench.config.config_run import ConfigRun
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

        self.cfg = cfg
        self.n_classes = n_classes
        self.feature_names = feature_names

    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):

        x_df = pd.DataFrame(x_train, columns=self.feature_names)
        x_df["__TARGET__"] = y_train

        self.predictor = TabularPredictor(label="__TARGET__").fit(x_df, time_limit=self.cfg.hyperparams['time_limit'])


    def test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> PredictionMetrics:

        x_df = pd.DataFrame(x_test, columns=self.feature_names)
        y_pred = self.predictor.predict(x_df)

        prediction_metrics = PredictionMetrics.from_prediction(y_pred, y_test, self.cfg.task)
        return prediction_metrics