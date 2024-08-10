import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.base import BaseEstimator

from tabularbench.config.config_run import ConfigRun
from tabularbench.core.enums import Task
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

        x_train_df = pd.DataFrame(x_train, columns=self.feature_names)
        x_train_df["__TARGET__"] = y_train

        x_tuning_df = pd.DataFrame(x_val, columns=self.feature_names)
        x_tuning_df["__TARGET__"] = y_val

        x_combined = pd.concat([x_train_df, x_tuning_df])

        path = Path("temp_autogluon") / f"device_{self.cfg.device}"
        if path.exists():
            # Autogluon writes all models to disk, which takes up enormous amounts of space
            # We delete the folder and recreate it to avoid running out of disk
            shutil.rmtree(path)
        path.mkdir(parents=True)

        # Autogluon does not correctly infer the problem type from the data on Openml Dataset 146024
        match self.cfg.task:
            case Task.CLASSIFICATION:
                problem_type = "multiclass"
            case Task.REGRESSION:
                problem_type = "regression"

        self.predictor = TabularPredictor(
            label="__TARGET__",
            problem_type=problem_type,
            path=path,
        ).fit(
            train_data=x_combined,
            time_limit=self.cfg.hyperparams['time_limit'],
            presets=['best_quality', 'optimize_for_deployment'],
            num_gpus=0,
            num_cpus=self.cfg.hyperparams['num_cpus'],
        )


    def test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> PredictionMetrics:

        x_df = pd.DataFrame(x_test, columns=self.feature_names)
        y_pred = self.predictor.predict_proba(x_df).values

        prediction_metrics = PredictionMetrics.from_prediction(y_pred, y_test, self.cfg.task)
        return prediction_metrics