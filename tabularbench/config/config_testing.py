from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf

from tabularbench.core.enums import BenchmarkName, DownstreamTask


@dataclass
class ConfigTesting():
    downstream_tasks: list[DownstreamTask]
    n_default_runs_per_dataset_valid: int
    n_default_runs_per_dataset_test: int               
    openml_dataset_ids_to_ignore: list[int]
    benchmarks: list[BenchmarkName]

    @classmethod
    def from_hydra(cls, cfg_hydra: DictConfig):
        
        downstream_tasks = [DownstreamTask[downstream_task] for downstream_task in cfg_hydra.downstream_tasks]
        benchmarks = [BenchmarkName[benchmark] for benchmark in cfg_hydra.benchmarks]

        return cls(
            downstream_tasks=downstream_tasks,
            n_default_runs_per_dataset_valid=cfg_hydra.n_default_runs_per_dataset_valid,
            n_default_runs_per_dataset_test=cfg_hydra.n_default_runs_per_dataset_test,
            openml_dataset_ids_to_ignore=OmegaConf.to_container(cfg_hydra.openml_dataset_ids_to_ignore),    # type: ignore   
            benchmarks=benchmarks
        )