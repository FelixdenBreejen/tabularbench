from __future__ import annotations

import pandas as pd


from tabularbench.core.enums import SearchType
from tabularbench.results.dataset_plot_combined import make_combined_dataset_plot
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.paths_and_filenames import (
    DEFAULT_RESULTS_FILE_NAME
)
from tabularbench.results.default_results import make_default_results
from tabularbench.results.hyperparam_plots import make_hyperparam_plots


def plot_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    if len(df_run_results) == 0:
        # no results yet to plot
        return

    cfg.logger.info(f"Start making plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")

    if sweep_default_finished(cfg, df_run_results) and default_results_not_yet_made(cfg):
        cfg.logger.info(f"Start making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")
        make_default_results(cfg, df_run_results)
        cfg.logger.info(f"Finished making default results for model {cfg.model_name.value} on benchmark {cfg.benchmark.name}")

    
    if cfg.search_type == SearchType.RANDOM:
        cfg.logger.info(f"Start making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
        make_hyperparam_plots(cfg, df_run_results)
        cfg.logger.info(f"Finished making hyperparam plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    

    cfg.logger.info(f"Start making sweep plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    make_combined_dataset_plot(cfg, df_run_results)
    # make_separate_dataset_plots(cfg, df_run_results)
    cfg.logger.info(f"Finished making sweep plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")
    
    cfg.logger.info(f"Finished making plots for {cfg.search_type.value} search for {cfg.model_name.value} on {cfg.benchmark.name}")



def sweep_default_finished(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    df = df_run_results
    df = df[ df['search_type'] == SearchType.DEFAULT.name ]
    df = df[ df['seed'] == cfg.seed ]    # when using multiple default runs, the seed changes

    for dataset_id in cfg.openml_dataset_ids_to_use:

        df_id = df[ df['openml_dataset_id'] == dataset_id ]
        if len(df_id) == 0:
            return False

    return True


def default_results_not_yet_made(cfg: ConfigBenchmarkSweep) -> bool:
    return not (cfg.output_dir / DEFAULT_RESULTS_FILE_NAME).exists()


