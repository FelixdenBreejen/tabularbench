import numpy as np
from matplotlib import pyplot as plt

from tabularbench.config.config_benchmark_sweep import ConfigBenchmarkSweep
from tabularbench.results.random_sequence import normalize_sequences


def make_combined_dataset_plot_data(cfg: ConfigBenchmarkSweep, sequences_all: np.ndarray) -> np.ndarray:

    sequences_all = normalize_sequences(cfg, sequences_all) # [models, datasets, shuffles, runs]
    sequences_all = np.mean(sequences_all, axis=1)   # [models, sequence_length, n_shuffles]

    n_models = sequences_all.shape[0]
    
    plot_data = np.empty((3, n_models, cfg.plotting.whytrees.n_runs))

    for model_i in range(n_models):

        sequences_model = sequences_all[model_i, :, :]

        sequence_mean = np.mean(sequences_model, axis=0)
        sequence_lower_bound = np.quantile(sequences_model, q=1-cfg.plotting.whytrees.confidence_bound, axis=0)
        sequence_upper_bound = np.quantile(sequences_model, q=cfg.plotting.whytrees.confidence_bound, axis=0)

        plot_data[0, model_i, :] = sequence_mean
        plot_data[1, model_i, :] = sequence_lower_bound
        plot_data[2, model_i, :] = sequence_upper_bound

    return plot_data


def make_combined_dataset_plot(cfg: ConfigBenchmarkSweep, plot_data: np.ndarray) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(25, 25))

    models = cfg.plotting.whytrees.benchmark_model_names + [cfg.model_plot_name]

    for model_i, model in enumerate(models):

        sequence_mean = plot_data[0, model_i, :]
        sequence_lower_bound = plot_data[1, model_i, :]
        sequence_upper_bound = plot_data[2, model_i, :]

        epochs = np.arange(len(sequence_mean)) + cfg.plotting.whytrees.plot_default_value

        ax.plot(epochs, sequence_mean, label=model, linewidth=12)
        ax.fill_between(x=epochs, y1=sequence_lower_bound, y2=sequence_upper_bound, alpha=0.2)


    ax.set_title(f"Averaged Normalized Test Score \n for all datasets of benchmark {cfg.benchmark.name}", fontsize=50)
    ax.set_xlabel("Number of runs", fontsize=50)
    ax.set_ylabel("Normalized Test score", fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=40)

    ax.set_xscale('log')
    ax.set_xlim([1, cfg.plotting.whytrees.n_runs])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=40, handlelength=3)
    fig.tight_layout(pad=2.0, rect=[0, 0.12, 1, 0.98])

    return fig














