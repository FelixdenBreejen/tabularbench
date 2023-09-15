from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from omegaconf import DictConfig
import itertools
import logging

import torch
import openml
import pandas as pd

from tabularbench.data.benchmarks import benchmarks, benchmark_names
from tabularbench.sweeps.writer import Writer
from tabularbench.core.enums import DatasetSize, Task, FeatureType, SearchType


@dataclass
class SweepConfig():
    logger: logging.Logger
    writer: Writer
    output_dir: Path
    seed: int
    device: torch.device
    model: str
    model_plot_name: str
    task: Task
    feature_type: FeatureType
    benchmark_name: str
    search_type: SearchType
    openml_suite_id: int
    openml_task_ids: list[int]
    openml_dataset_ids: list[int]
    openml_dataset_names: list[str]
    runs_per_dataset: int
    dataset_size: DatasetSize
    hyperparams: DictConfig      # hyperparameters for the model

    def __post_init__(self):

        assert self.benchmark_name in benchmark_names, f"{self.benchmark_name} is not a valid benchmark. Please choose from {benchmark_names}"
        self.sweep_dir = self.output_dir / f'{self.benchmark_name}_{self.model}_{self.search_type}'



    @classmethod
    def from_dict(cls, sweep_dict: dict) -> SweepConfig:

        return cls(
            model=sweep_dict['model'],
            plot_name=sweep_dict['plot_name'],
            task=sweep_dict['task'],
            feature_type=sweep_dict['feature_type'],
            benchmark_name=sweep_dict['benchmark_name'],
            search_type=sweep_dict['search_type'],
            runs_per_dataset=sweep_dict['runs_per_dataset'],
            dataset_size=sweep_dict['dataset_size'],
        )


    def __str__(self) -> str:

        return f"{self.benchmark_name}-{self.search_type}-{self.model}"
    

def create_sweep_config_list_from_main_config(cfg: DictConfig, writer: Writer, logger: logging.Logger) -> list[SweepConfig]:

    sweep_configs = []

    assert len(cfg.models) == len(cfg.model_plot_names), f"Please provide a plot name for each model. Got {len(cfg.models)} models and {len(cfg.model_plot_names)} plot names."

    models_with_plot_name = zip(cfg.models, cfg.model_plot_names)
    sweep_details = itertools.product(models_with_plot_name, cfg.search_type, cfg.benchmarks)

    for (model, model_plot_name), search_type, benchmark_name in sweep_details:

        for b in benchmarks:
            if b['name'] == benchmark_name:
                benchmark = b
                break
        else:
            raise ValueError(f"Benchmark {benchmark_name} not found in benchmarks")
        
        if hasattr(cfg, 'device'):
            device = torch.device(cfg.device)
        else:
            device = None

        if search_type == 'default':
            runs_per_dataset = 1
        else:
            runs_per_dataset = cfg.runs_per_dataset

        if benchmark['categorical']:
            feature_type = FeatureType.MIXED
        else:
            feature_type = FeatureType.NUMERICAL

        if benchmark['task'] == 'regression':
            task = Task.REGRESSION
        elif benchmark['task'] == 'classif':
            task = Task.CLASSIFICATION
        else:
            raise ValueError(f"task must be one of ['regression', 'classif']. Got {benchmark['task']}")

            
        if benchmark['dataset_size'] == 'small':
            dataset_size = DatasetSize.SMALL
        elif benchmark['dataset_size'] == 'medium':
            dataset_size = DatasetSize.MEDIUM
        elif benchmark['dataset_size'] == 'large':
            dataset_size = DatasetSize.LARGE
        else:
            raise ValueError(f"dataset_size_str must be one of ['small', 'medium', 'large']. Got {benchmark['dataset_size']}")
        
        assert model in cfg.hyperparams, f"Model {model} not found in main configuration's hyperparams"

        openml_suite = openml.study.get_suite(benchmark['suite_id'])
        openml_task_ids = openml_suite.tasks
        openml_dataset_ids = openml_suite.data
        openml_dataset_names = []
        
        for dataset_id in openml_dataset_ids:
            dataset = openml.datasets.get_dataset(dataset_id, download_data=False, download_qualities=False, download_features_meta_data=False)
            openml_dataset_names.append(dataset.name)

        sweep_config = SweepConfig(
            logger=logger,
            writer=writer,
            output_dir=Path(cfg.output_dir),
            seed=cfg.seed,
            device=device,
            model=model,
            model_plot_name=model_plot_name,
            benchmark_name=benchmark['name'],
            search_type=search_type,
            task=task,
            dataset_size=dataset_size,
            feature_type=feature_type,
            openml_suite_id=benchmark['suite_id'],
            openml_task_ids=openml_task_ids,
            openml_dataset_ids=openml_dataset_ids,
            openml_dataset_names=openml_dataset_names,
            runs_per_dataset=runs_per_dataset,
            hyperparams=cfg.hyperparams[model]
        )

        sweep_configs.append(sweep_config)

    return sweep_configs
