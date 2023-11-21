from __future__ import annotations

import pandas as pd

from tabularbench.core.enums import ModelName, SearchType
from tabularbench.results.reformat_benchmark import get_benchmark_csv_reformatted
from tabularbench.sweeps.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.sweeps.paths_and_filenames import (
    DEFAULT_RESULTS_FILE_NAME
)


def make_default_results(cfg: ConfigBenchmarkSweep, df_run_results: pd.DataFrame) -> None:

    benchmark_model_names = [model_name.name for model_name in cfg.config_plotting.benchmark_model_names]

    df_bench = get_benchmark_csv_reformatted()
    df_bench = df_bench[ df_bench['openml_dataset_id'].isin(cfg.openml_dataset_ids_to_use) ]
    df_bench = df_bench[ df_bench['model'].isin(benchmark_model_names) ]
    df_bench = df_bench[ df_bench['search_type'] == SearchType.DEFAULT.name ]
    df_bench['model_plot_name'] = df_bench.apply(lambda row: ModelName[row['model']].value, axis=1)
    df_bench.sort_values(by=['model', 'openml_dataset_id'], inplace=True)

    df_run_results = df_run_results[ df_run_results['search_type'] == SearchType.DEFAULT.name ]
    df_run_results = df_run_results[ df_run_results['seed'] == cfg.seed ] # when using multiple default runs, the seed changes
    df_run_results['model_plot_name'] = cfg.model_plot_name
    df_run_results.sort_values(by=['openml_dataset_id'], inplace=True)

    df = pd.concat([df_bench, df_run_results], ignore_index=True)
    df['openml_dataset_name'] = df.apply(lambda row: row['openml_dataset_name'][:8] + '...' if len(row['openml_dataset_name']) > 11 else row['openml_dataset_name'], axis=1)
    df['score_test_mean'] = df['score_test_mean'].apply(lambda x: f"{x:.4f}")

    df_results = df.pivot(index='model', columns=['openml_dataset_id', 'openml_dataset_name'], values='score_test_mean')
    df_results.to_csv(cfg.output_dir / DEFAULT_RESULTS_FILE_NAME, mode='w', index=True, header=True)