import torch

from tabularbench.config.config_run import ConfigRun
from tabularbench.core.enums import ModelName
from tabularbench.core.trainer import Trainer
from tabularbench.core.trainer_finetune import TrainerFinetune
from tabularbench.models.autogluon.trainer import Trainer as TrainerAutoGluon
from tabularbench.models.carte.trainer import Trainer as TrainerCarte


def get_trainer(cfg: ConfigRun, model: torch.nn.Module, n_classes: int, feature_names: list[str]):

    match cfg.model_name:
        case ModelName.FT_TRANSFORMER:
            return Trainer(cfg, model, n_classes)
        case ModelName.CARTE:
            return TrainerCarte(cfg, model, n_classes, feature_names)
        case ModelName.AUTOGLUON:
            return TrainerAutoGluon(cfg, model, n_classes, feature_names)
        case ModelName.TABPFN | ModelName.FOUNDATION:
            return TrainerFinetune(cfg, model, n_classes)
        case _:
            raise NotImplementedError(f"Model {cfg.model_name} not implemented")