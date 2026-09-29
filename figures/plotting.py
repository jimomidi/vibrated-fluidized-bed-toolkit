"""Reusable, publication-style plotting functions for parameter-sweep results.

The original exploratory analysis produced a large number of near-identical
Matplotlib scripts -- one per figure -- that differed only in the input
numbers, axis labels, and minor cosmetic tweaks. This module collapses that
into a small set of parameterized, reusable plotting functions covering
every distinct figure *type* that appeared:

- :func:`plot_errorbar_by_case` -- mean +/- std vs. a categorical case axis.
- :func:`plot_dual_axis_errorbar` -- the same, with two related case axes
  (e.g. frequency ratio and vibration strength) sharing one y-axis.
- :func:`plot_case_time_heatmap` -- a case x time matrix rendered as a heatmap.
- :func:`plot_joint_scatter_with_marginals` -- a scatter plot with marginal
  histograms/KDEs for one or more labeled datasets.
- :func:`plot_boxplot_with_jitter` -- a horizontal boxplot overlaid with
  jittered raw points, one box per row of a dataframe.
- :func:`plot_multi_series_timeseries` -- several labeled time series on one
  axis (e.g. a mixing index over time for several cases).
- :func:`plot_striped_time_evolution` -- several offset error-bar series
  across discrete time points, with alternating background bands.
- :func:`plot_grouped_bar_with_error` -- grouped bars with asymmetric error
  bars (e.g. simulation vs. experiment comparison).

Every function accepts plain arrays/dataframes and returns the created
``Figure``/``Axes`` so callers can further customize or save the result;
none of them read files or embed study-specific numbers.

The ``__main__`` block at the bottom renders every function once with
synthetic placeholder data purely to demonstrate the API -- it does not
represent real measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde


def apply_default_style(font_family: str = "Times New Roman") -> None:
    """Apply the bold, serif publication style used throughout the original figures."""
    plt.rcParams["font.family"] = font_family
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["legend.frameon"] = False


def _maybe_save(fig: plt.Figure, output_path: Path | str | None) -> None:
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_errorbar_by_case(
    case_labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    xlabel: str,
    ylabel: str = "",
    figsize: tuple[float, float] = (6, 3),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Mean +/- std markers at evenly spaced categorical case positions."""
    fig, ax = plt.subplots(figsize=figsize)
    x_positions = range(len(case_labels))

    ax.errorbar(
        x_positions, means, yerr=stds, fmt="o", capsize=5, markersize=14,
        linestyle="none", color="black", ecolor="black", elinewidth=1, markeredgewidth=1.5,
    )
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=15)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(case_labels)
    ax.grid(False)
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


def plot_dual_axis_errorbar(
    bottom_labels: Sequence[str],
    top_labels: Sequence[str],
    means_bottom: Sequence[float],
    stds_bottom: Sequence[float],
    means_top: Sequence[float],
    stds_top: Sequence[float],
    bottom_xlabel: str,
    top_xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (6, 4),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Two error-bar series sharing one y-axis, each keyed to its own x-axis (bottom/top)."""
    fig, ax_bottom = plt.subplots(figsize=figsize)
    x_idx = np.arange(len(bottom_labels))

    ax_bottom.errorbar(x_idx, means_bottom, yerr=stds_bottom, fmt="o--", color="blue", capsize=4, markersize=8, label=bottom_xlabel)
    ax_bottom.set_xticks(x_idx)
    ax_bottom.set_xticklabels(bottom_labels, color="blue")
    ax_bottom.set_xlabel(bottom_xlabel, fontweight="bold", labelpad=15, color="blue")

    ax_top = ax_bottom.twiny()
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.errorbar(x_idx, means_top, yerr=stds_top, fmt="o--", color="red", capsize=4, markersize=8, label=top_xlabel)
    ax_top.set_xticks(x_idx)
    ax_top.set_xticklabels(top_labels, color="red")
    ax_top.set_xlabel(top_xlabel, fontweight="bold", labelpad=15, color="red")

    ax_bottom.set_ylabel(ylabel, fontweight="bold", labelpad=15)
    ax_bottom.legend(loc="best")
    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax_bottom


def plot_case_time_heatmap(
    matrix: np.ndarray | pd.DataFrame,
    case_labels: Sequence[str],
    time_labels: Sequence[str],
    ylabel: str,
    colorbar_label: str,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] = (10, 4),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Render a (case x time) matrix as a heatmap with a labeled colorbar."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=90, ha="center")
    ax.set_yticks(np.arange(len(case_labels)))
    ax.set_yticklabels(case_labels)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=15)

    colorbar = fig.colorbar(im, ax=ax, pad=0.01)
    colorbar.set_label(colorbar_label, labelpad=15, fontweight="bold")

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


def plot_joint_scatter_with_marginals(
    datasets: Sequence[tuple[np.ndarray, np.ndarray, str, str]],
    xlabel: str,
    ylabel: str,
    show_kde: bool = True,
    density: bool = True,
    n_bins: int = 25,
    figsize: tuple[float, float] = (7, 7),
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Scatter plot with marginal histograms (and optional KDE curves) for labeled datasets.

    ``datasets`` is a sequence of ``(x, y, label, color)`` tuples, one per
    group to overlay (e.g. two different parameter sweeps).
    """
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(2, 2, width_ratios=[7, 2], height_ratios=[2, 7], wspace=0.05, hspace=0.05)
    ax_joint = fig.add_subplot(grid[1, 0])
    ax_marg_x = fig.add_subplot(grid[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(grid[1, 1], sharey=ax_joint)

    all_x = np.concatenate([d[0] for d in datasets])
    all_y = np.concatenate([d[1] for d in datasets])
    bins_x = np.linspace(all_x.min(), all_x.max(), n_bins)
    bins_y = np.linspace(all_y.min(), all_y.max(), n_bins)

    legend_elements = []
    for x, y, label, color in datasets:
        ax_joint.scatter(x, y, color=color, alpha=0.8, label=label, s=80)
        ax_marg_x.hist(x, bins=bins_x, color=color, alpha=0.8, density=density, edgecolor="white", linewidth=0.5)
        ax_marg_y.hist(y, bins=bins_y, color=color, alpha=0.8, density=density, orientation="horizontal", edgecolor="white", linewidth=0.5)

        if show_kde and x.size > 1 and y.size > 1:
            try:
                x_vals = np.linspace(bins_x[0], bins_x[-1], 200)
                ax_marg_x.plot(x_vals, gaussian_kde(x)(x_vals), color=color, linewidth=2)
                y_vals = np.linspace(bins_y[0], bins_y[-1], 200)
                ax_marg_y.plot(gaussian_kde(y)(y_vals), y_vals, color=color, linewidth=2)
            except np.linalg.LinAlgError:
                pass  # KDE is undefined for degenerate (zero-variance) samples.

        legend_elements.append(
            Line2D([0], [0], marker="s", color="w", label=label, markerfacecolor=color, markersize=15, markeredgecolor=color)
        )

    ax_joint.set_xlabel(xlabel, fontweight="bold", fontsize=18, labelpad=10)
    ax_joint.set_ylabel(ylabel, fontweight="bold", fontsize=18, labelpad=10)
    ax_joint.tick_params(labelsize=20)
    ax_joint.legend(handles=legend_elements, loc="upper right", fontsize=15, frameon=False)

    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig


def plot_boxplot_with_jitter(
    data: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    box_color: str = "#fffacd",
    figsize: tuple[float, float] = (6, 3),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal boxplot (one box per row of ``data``) with jittered raw points overlaid."""
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=data.T, orient="h", whis=1.5, showcaps=True, ax=ax,
        boxprops={"facecolor": box_color, "edgecolor": "black"},
        showmeans=True, meanline=True, meanprops={"color": "green", "linewidth": 1},
    )

    rng = np.random.default_rng()
    for row_index in range(data.shape[0]):
        jitter = rng.uniform(-0.2, 0.2, size=data.shape[1])
        ax.scatter(data.iloc[row_index], row_index + jitter, color="black", alpha=0.7, zorder=5)

    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels([str(i) for i in range(data.shape[0])])
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=10)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


def plot_multi_series_timeseries(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    colors: Sequence[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (6.5, 3.5),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay several labeled ``(x, y)`` time series (e.g. mixing index vs. time per case)."""
    fig, ax = plt.subplots(figsize=figsize)
    palette = colors or plt.cm.tab10(np.linspace(0, 1, len(series)))

    for (label, (x, y)), color in zip(series.items(), palette):
        ax.plot(x, y, label=label, color=color)

    ax.set_xlabel(xlabel, fontweight="bold", labelpad=12)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=12)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.05))

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


def plot_striped_time_evolution(
    series: Mapping[str, tuple[np.ndarray, np.ndarray]],
    x_tick_labels: Sequence[str],
    xlabel: str,
    ylabel: str,
    offset_step: float = 0.07,
    figsize: tuple[float, float] = (11, 4),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Several offset mean+/-std series across discrete points, on an alternating-band background.

    ``series`` maps a case label to ``(means, stds)`` arrays, one value per
    entry of ``x_tick_labels``. Each case's markers are horizontally offset
    slightly so overlapping error bars remain legible.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x_base = np.arange(len(x_tick_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(series)))

    for i, (label, (means, stds)) in enumerate(series.items()):
        x_shifted = x_base + (i - len(series) / 2) * offset_step
        ax.errorbar(x_shifted, means, yerr=stds, fmt="-o", color=colors[i], capsize=3, linewidth=1.3, markersize=8, label=label)

    for i in range(len(x_base)):
        color = "#d9d9d9" if i % 2 == 0 else "white"
        ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.8, zorder=0)

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_tick_labels, fontweight="bold")
    ax.set_xlim(-0.5, len(x_base) - 0.5)
    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold", labelpad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, labelspacing=0.3)
    ax.grid(False)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


def plot_grouped_bar_with_error(
    x: np.ndarray,
    group_a: tuple[np.ndarray, np.ndarray, str],
    group_b: tuple[np.ndarray, np.ndarray, str],
    x_tick_labels: Sequence[str],
    ylabel: str,
    bar_width: float = 0.3,
    colors: tuple[str, str] = ("#B3E5FC", "#F4A7A7"),
    figsize: tuple[float, float] = (5, 4),
    output_path: Path | str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart (e.g. simulation vs. experiment) with symmetric error bars.

    ``group_a``/``group_b`` are ``(values, std_errors, legend_label)`` tuples.
    """
    fig, ax = plt.subplots(figsize=figsize)
    offset = bar_width / 2
    x_a, x_b = x - offset, x + offset

    for x_group, (values, errors, label), color in ((x_a, group_a, colors[0]), (x_b, group_b, colors[1])):
        ax.bar(x_group, values, width=bar_width, label=label, color=color)
        cap_width = bar_width * 0.6
        for xi, value, error in zip(x_group, values, errors):
            ax.vlines(x=xi, ymin=value - error, ymax=value + error, color="black", linewidth=0.8)
            ax.hlines(y=value + error, xmin=xi - cap_width / 2, xmax=xi + cap_width / 2, color="black", linewidth=0.8)
            ax.hlines(y=value - error, xmin=xi - cap_width / 2, xmax=xi + cap_width / 2, color="black", linewidth=0.8)

    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold", labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels, fontsize=15, rotation=45)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=14, frameon=False)

    fig.tight_layout()
    _maybe_save(fig, output_path)
    return fig, ax


if __name__ == "__main__":
    # Demonstration only: every function below is called with synthetic,
    # randomly generated placeholder data so the module can be smoke-tested
    # without any real dataset. None of these numbers represent measurements.
    apply_default_style()
    rng = np.random.default_rng(seed=0)
    demo_cases = ["0", "0.25", "0.5", "0.75", "1", "1.5", "2", "2.5"]

    plot_errorbar_by_case(demo_cases, rng.normal(size=8), rng.uniform(0.1, 0.3, size=8), xlabel="Example Case")

    plot_case_time_heatmap(
        rng.uniform(size=(8, 10)), demo_cases, [f"t{i}" for i in range(10)],
        ylabel="Example Case", colorbar_label="Example Value",
    )

    plot_joint_scatter_with_marginals(
        [
            (rng.normal(size=100), rng.normal(size=100), "Group A", "#053061"),
            (rng.normal(size=100), rng.normal(size=100), "Group B", "#67001f"),
        ],
        xlabel="Example X", ylabel="Example Y",
    )

    plot_boxplot_with_jitter(pd.DataFrame(rng.normal(size=(8, 15))), xlabel="Example Ratio", ylabel="Example Case")

    plot_multi_series_timeseries(
        {label: (np.linspace(0, 5, 50), rng.uniform(0.4, 1.0, 50)) for label in demo_cases[:3]},
        xlabel="Time (s)", ylabel="Example Index",
    )

    plot_striped_time_evolution(
        {label: (rng.uniform(size=9), rng.uniform(0.01, 0.05, size=9)) for label in demo_cases[:3]},
        x_tick_labels=[f"n{i + 1}" for i in range(9)], xlabel="Time", ylabel="Example Measurement",
    )

    plot_grouped_bar_with_error(
        np.array([0.8, 1.6, 2.4, 3.2]),
        (rng.uniform(10, 50, 4), rng.uniform(1, 4, 4), "Computational Results"),
        (rng.uniform(10, 50, 4), rng.uniform(1, 4, 4), "Experimental Data"),
        x_tick_labels=["A", "B", "C", "D"], ylabel="Example Metric",
    )

    plt.show()
