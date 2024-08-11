from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self

import torch

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.config.config_save_load_mixin import ConfigSaveLoadMixin
from tabularbench.core.enums import DatasetSize, DownstreamTask, ModelName, Task
from tabularbench.data.datafile_openml import OpenmlDatafile


@dataclass
class ConfigRun(ConfigSaveLoadMixin):
    output_dir: Path
    device: torch.device
    cpus: Optional[list[int]]
    seed: int
    model_name: ModelName
    task: Task
    dataset_size: Optional[DatasetSize]
    openml_dataset_id: int
    openml_dataset_name: str
    datafile_path: Path
    hyperparams: dict


    @classmethod
    def create(
        cls, 
        cfg: ConfigBenchmarkSweep, 
        seed: int,
        device: torch.device, 
        dataset_file_path: Path,
        hyperparams: dict,
        run_id: int
    ) -> Self:

        dataset_size = cfg.benchmark.dataset_size
        openml_datafile = OpenmlDatafile(dataset_file_path)
        openml_dataset_id = openml_datafile.ds.attrs['openml_dataset_id']
        openml_dataset_name = openml_datafile.ds.attrs['openml_dataset_name']
        
        output_dir = cfg.output_dir / str(openml_dataset_id) / f"#{run_id}"

        if cfg.downstream_task == DownstreamTask.ZEROSHOT:
            hyperparams['max_epochs'] = 0

        if cfg.max_cpus_per_device is not None:
            cpus = [device.index * cfg.max_cpus_per_device + i for i in range(cfg.max_cpus_per_device)]
        else:
            cpus = None

        return cls(
            output_dir=output_dir,
            model_name=cfg.model_name,
            device=device,
            cpus=cpus,
            seed=seed,
            task=cfg.benchmark.task,
            dataset_size=dataset_size,
            openml_dataset_id=openml_dataset_id,
            openml_dataset_name=openml_dataset_name,
            datafile_path=dataset_file_path,
            hyperparams=hyperparams
        )

            