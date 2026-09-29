"""Temporal autocorrelation of void fraction, used to estimate wave periodicity.

For a fixed time lag, this module compares the gas void-fraction field
(``EP_G``) of an unstructured CFD grid (``.vtu``) at time ``t`` against the
same field at time ``t + lag``, restricted to a horizontal band of the
domain. The Pearson correlation coefficient of the two fields is computed
for every valid pair of snapshots in a run; a high, stable correlation at a
given lag indicates that the flow structure (e.g. a standing wave pattern)
repeats with that period.

Because a handful of snapshot pairs can be corrupted by transient numerical
noise, the per-case correlation coefficients are cleaned with a
multi-stage robust filter (median-absolute-deviation, Tukey IQR fallback,
and a rolling Hampel filter) before the mean and standard deviation are
reported.

Expected input layout
----------------------
    <root_dir>/
        <case_1>/*.vtu
        <case_2>/*.vtu
        ...

Each ``.vtu`` file must expose the configured cell-data variable (default:
``EP_G``).

Usage
-----
    python -m analysis.void_fraction_correlation \\
        --root-dir data/void_fraction/vibration_strength_sweep \\
        --delta-t 14 --variable EP_G --y-bounds 0.015 0.035 \\
        --output-csv results/void_fraction_correlation.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import vtk

_NUMBER_PATTERN = re.compile(r"(\d+)")

# Robust-filtering defaults, tuned empirically on the original dataset.
DEFAULT_MAD_THRESHOLD_R = 2.5  # MAD threshold applied to the correlation coefficients
DEFAULT_MAD_THRESHOLD_D = 3.0  # MAD threshold applied to the z-score distance metric
DEFAULT_TUKEY_K = 1.5  # IQR multiplier, used as a fallback when MAD collapses to 0
DEFAULT_HAMPEL_K = 3.0  # MAD multiplier for the rolling Hampel filter
DEFAULT_HAMPEL_WINDOW = 11  # window size for the rolling Hampel filter (must be odd)
DEFAULT_MIN_SHARED_CELLS = 50  # minimum overlapping cell count to trust a pair
DEFAULT_MIN_RESULTS_FOR_FILTER = 7  # below this sample count, skip the hard filters


def read_unstructured_grid(filepath: Path):
    """Read a VTK unstructured grid (``.vtu``) file."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    grid = reader.GetOutput()
    if not grid or grid.GetNumberOfCells() == 0:
        raise ValueError(f"Empty or unreadable grid: {filepath}")
    return grid


def extract_cell_variable_in_band(grid, variable_name: str, y_bounds: tuple[float, float]) -> np.ndarray:
    """Return the values of a cell-data variable for cells whose center falls within a y-band."""
    array = grid.GetCellData().GetArray(variable_name)
    if array is None:
        raise ValueError(f"Variable '{variable_name}' not found in cell data.")

    values = []
    for i in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(i)
        points = cell.GetPoints()
        center_y = np.mean([points.GetPoint(j)[1] for j in range(points.GetNumberOfPoints())])
        if y_bounds[0] < center_y < y_bounds[1]:
            values.append(array.GetValue(i))

    return np.array(values, dtype=float)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient; returns NaN for degenerate inputs."""
    if x.size < 2 or y.size < 2:
        return np.nan
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denominator) if denominator != 0 else np.nan


def zscore(values: np.ndarray) -> np.ndarray | None:
    std = np.std(values)
    if std == 0:
        return None
    return (values - np.mean(values)) / std


def mad_mask(values: np.ndarray, k: float, side: str = "both") -> np.ndarray:
    """Median-absolute-deviation keep-mask. ``side`` is 'both', 'upper', or 'lower'."""
    values = np.asarray(values)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return np.ones_like(values, dtype=bool)

    modified_z = 0.6745 * (values - median) / mad
    if side == "both":
        return np.abs(modified_z) <= k
    if side == "upper":
        return modified_z <= k
    if side == "lower":
        return modified_z >= -k
    raise ValueError(f"Unknown side: {side}")


def tukey_mask(values: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Interquartile-range keep-mask."""
    values = np.asarray(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values >= q1 - k * iqr) & (values <= q3 + k * iqr)


def hampel_mask(series: np.ndarray, window: int = 11, k: float = 3.0) -> np.ndarray:
    """Rolling Hampel filter keep-mask, for removing localized spikes in a sequence."""
    values = np.asarray(series, dtype=float)
    n = len(values)
    keep = np.ones(n, dtype=bool)
    if n < window or window < 3 or window % 2 == 0:
        return keep

    half = window // 2
    for i in range(n):
        segment = values[max(0, i - half): min(n, i + half + 1)]
        median = np.nanmedian(segment)
        mad = np.nanmedian(np.abs(segment - median))
        if mad == 0:
            continue
        if np.abs(values[i] - median) > k * 1.4826 * mad:
            keep[i] = False
    return keep


@dataclass
class CorrelationStatistics:
    case: str
    mean_r: float
    std_r: float
    kept: int
    total: int


def robust_filtered_correlation_statistics(
    r_values: np.ndarray,
    distance_values: np.ndarray,
    mad_threshold_r: float = DEFAULT_MAD_THRESHOLD_R,
    mad_threshold_d: float = DEFAULT_MAD_THRESHOLD_D,
    tukey_k: float = DEFAULT_TUKEY_K,
    hampel_k: float = DEFAULT_HAMPEL_K,
    hampel_window: int = DEFAULT_HAMPEL_WINDOW,
    min_results_for_filter: int = DEFAULT_MIN_RESULTS_FOR_FILTER,
) -> tuple[float, float, np.ndarray]:
    """Apply the multi-stage robust filter and return (mean, std, keep_mask).

    Stages: two-sided MAD filter on the correlation coefficients, an
    upper-sided MAD filter on the z-score distance between the two fields
    (large distance indicates a spurious/anomalous pair), a Tukey IQR
    fallback if the MAD filter on ``r`` cannot discriminate (MAD == 0), and
    a rolling Hampel filter to remove localized spikes in the ``r``
    sequence. If the combined filter removes every point, it falls back to
    the distance-based MAD filter alone.
    """
    mask_r_mad = mad_mask(r_values, k=mad_threshold_r, side="both")
    mask_d_mad = mad_mask(distance_values, k=mad_threshold_d, side="upper")
    keep = mask_r_mad & mask_d_mad

    if mask_r_mad.all():
        keep &= tukey_mask(r_values, k=tukey_k)

    if len(r_values) >= min_results_for_filter:
        keep &= hampel_mask(r_values, window=hampel_window, k=hampel_k)

    kept_r = r_values[keep]
    if kept_r.size == 0:
        keep = mask_d_mad
        kept_r = r_values[keep]

    mean_r = float(np.mean(kept_r)) if kept_r.size else float("nan")
    std_r = float(np.std(kept_r, ddof=1)) if kept_r.size > 1 else 0.0
    return mean_r, std_r, keep


def _sorted_indexed_files(folder: Path, suffix: str) -> list[tuple[int, Path]]:
    indexed = []
    for filepath in folder.iterdir():
        if filepath.suffix == suffix:
            match = _NUMBER_PATTERN.search(filepath.stem)
            if match:
                indexed.append((int(match.group(1)), filepath))
    return sorted(indexed, key=lambda item: item[0])


def compute_case_correlation(
    case_dir: Path,
    delta_t: int,
    variable_name: str,
    y_bounds: tuple[float, float],
    min_shared_cells: int = DEFAULT_MIN_SHARED_CELLS,
) -> CorrelationStatistics | None:
    """Compute the lag-``delta_t`` void-fraction correlation statistics for one case."""
    files = _sorted_indexed_files(case_dir, suffix=".vtu")

    r_values, distance_values = [], []
    for i in range(len(files) - delta_t):
        _, path_a = files[i]
        _, path_b = files[i + delta_t]
        try:
            values_a = extract_cell_variable_in_band(read_unstructured_grid(path_a), variable_name, y_bounds)
            values_b = extract_cell_variable_in_band(read_unstructured_grid(path_b), variable_name, y_bounds)
        except (ValueError, FileNotFoundError):
            continue

        n = min(values_a.size, values_b.size)
        if n < min_shared_cells:
            continue

        z_a, z_b = zscore(values_a[:n]), zscore(values_b[:n])
        if z_a is None or z_b is None:
            continue

        r = pearson_correlation(values_a[:n], values_b[:n])
        if np.isnan(r):
            continue

        r_values.append(r)
        distance_values.append(float(np.sqrt(np.mean((z_a - z_b) ** 2))))

    if not r_values:
        return None

    r_array, d_array = np.array(r_values), np.array(distance_values)
    mean_r, std_r, keep = robust_filtered_correlation_statistics(r_array, d_array)
    return CorrelationStatistics(case=case_dir.name, mean_r=mean_r, std_r=std_r, kept=int(keep.sum()), total=len(r_array))


def compute_sweep_correlation(
    root_dir: Path,
    delta_t: int,
    variable_name: str,
    y_bounds: tuple[float, float],
    min_shared_cells: int = DEFAULT_MIN_SHARED_CELLS,
) -> pd.DataFrame:
    """Compute robust lag correlation statistics for every numerically-named case sub-folder."""
    rows = []
    for case_dir in sorted(root_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.isdigit():
            continue

        stats = compute_case_correlation(case_dir, delta_t, variable_name, y_bounds, min_shared_cells=min_shared_cells)
        if stats is None:
            print(f"{case_dir.name}: no valid data")
            continue

        rows.append(stats)
        print(
            f"{stats.case}: mean r = {stats.mean_r:.6f}, std = {stats.std_r:.6f} "
            f"(kept {stats.kept}/{stats.total})"
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", type=Path, required=True, help="Parent directory containing one sub-folder per case.")
    parser.add_argument("--delta-t", type=int, required=True, help="Snapshot lag (in snapshot count, not seconds).")
    parser.add_argument("--variable", default="EP_G", help="Cell-data variable to correlate (default: void fraction, EP_G).")
    parser.add_argument("--y-bounds", type=float, nargs=2, required=True, metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--min-shared-cells",
        type=int,
        default=DEFAULT_MIN_SHARED_CELLS,
        help="Minimum overlapping cell count required to trust a snapshot pair.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = compute_sweep_correlation(
        args.root_dir, args.delta_t, args.variable, tuple(args.y_bounds), min_shared_cells=args.min_shared_cells
    )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved correlation summary to {args.output_csv}")


if __name__ == "__main__":
    main()
