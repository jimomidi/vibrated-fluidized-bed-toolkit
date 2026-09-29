"""Per-case statistics of particle contact force and velocity from VTP point clouds.

This module summarizes discrete-element (DEM) particle data exported as VTK
PolyData (``.vtp``) files -- one file per simulation snapshot, grouped into
one sub-folder per parameter case (e.g. a vibration-strength or
frequency-ratio sweep). For each case it aggregates the requested point-data
arrays across every snapshot and reports the mean, standard deviation, and
sample count.

Expected input layout
----------------------
    <input_dir>/
        <case_1>/*.vtp
        <case_2>/*.vtp
        ...

Expected point-data arrays (as written by the upstream CFD-DEM solver)
------------------------------------------------------------------------
    - Three scalar components of the contact-force vector (default names:
      ``User_Defined_Var_1/2/3``), combined into a magnitude and normalized
      by a user-supplied reference value (e.g. a characteristic particle
      weight) to obtain a dimensionless contact-force ratio.
    - A 3-component ``Velocity`` array, whose first (streamwise/horizontal)
      component is summarized directly.

Usage
-----
    python -m analysis.contact_force_velocity_stats \\
        --input-dir data/contact_force/frequency_ratio_sweep \\
        --cases 000 025 050 075 100 150 200 250 \\
        --variable contact_force \\
        --normalization 1.67e-7 \\
        --output-csv results/contact_force_by_case.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from vtk import vtkXMLPolyDataReader
from vtk.util.numpy_support import vtk_to_numpy

OutlierStrategy = Literal["none", "clip", "exclude"]

DEFAULT_CONTACT_FORCE_COMPONENTS = (
    "User_Defined_Var_1",
    "User_Defined_Var_2",
    "User_Defined_Var_3",
)
DEFAULT_VELOCITY_ARRAY = "Velocity"


@dataclass
class CaseStatistics:
    """Aggregated statistics for one parameter case."""

    case: str
    mean: float
    std: float
    count: int


def list_vtp_files(folder: Path) -> list[Path]:
    """Return the sorted list of ``.vtp`` files in ``folder``."""
    return sorted(folder.glob("*.vtp"))


def read_point_array(filepath: Path, array_name: str) -> np.ndarray:
    """Read a single named point-data array from a ``.vtp`` file.

    Works for both scalar arrays and multi-component (e.g. vector) arrays;
    the raw array is returned with shape ``(n_points,)`` or
    ``(n_points, n_components)``.
    """
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(filepath))
    reader.Update()
    dataset = reader.GetOutput()

    array = dataset.GetPointData().GetArray(array_name)
    if array is None:
        raise KeyError(f"Point-data array '{array_name}' not found in {filepath}")

    return vtk_to_numpy(array)


def compute_contact_force_magnitude(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, normalization: float
) -> np.ndarray:
    """Combine three orthogonal contact-force components into a normalized magnitude."""
    return np.sqrt(v1**2 + v2**2 + v3**2) / normalization


def apply_outlier_strategy(
    values: np.ndarray,
    strategy: OutlierStrategy,
    value_range: tuple[float, float] | None,
) -> np.ndarray:
    """Optionally clip or exclude values outside ``value_range``.

    The original exploratory analysis tried both approaches (capping
    extreme values vs. dropping them outright) at different points; both
    are kept here as explicit, selectable strategies rather than silently
    picking one.
    """
    if strategy == "none" or value_range is None:
        return values
    low, high = value_range
    if strategy == "clip":
        return np.clip(values, low, high)
    if strategy == "exclude":
        return values[(values >= low) & (values <= high)]
    raise ValueError(f"Unknown outlier strategy: {strategy}")


def compute_case_contact_force_statistics(
    case_dir: Path,
    component_arrays: Sequence[str] = DEFAULT_CONTACT_FORCE_COMPONENTS,
    normalization: float = 1.0,
    take_abs: bool = True,
    outlier_strategy: OutlierStrategy = "none",
    value_range: tuple[float, float] | None = None,
) -> CaseStatistics | None:
    """Aggregate normalized contact-force magnitude across all snapshots in a case."""
    all_values: list[np.ndarray] = []

    for filepath in list_vtp_files(case_dir):
        try:
            components = [read_point_array(filepath, name) for name in component_arrays]
        except KeyError:
            continue

        magnitude = compute_contact_force_magnitude(*components, normalization=normalization)
        if take_abs:
            magnitude = np.abs(magnitude)
        magnitude = apply_outlier_strategy(magnitude, outlier_strategy, value_range)

        if magnitude.size > 0:
            all_values.append(magnitude)

    if not all_values:
        return None

    combined = np.concatenate(all_values)
    return CaseStatistics(case=case_dir.name, mean=float(np.mean(combined)), std=float(np.std(combined)), count=combined.size)


def compute_case_velocity_statistics(
    case_dir: Path,
    velocity_array: str = DEFAULT_VELOCITY_ARRAY,
    component_index: int = 0,
) -> CaseStatistics | None:
    """Aggregate one velocity component (default: horizontal/x) across all snapshots in a case."""
    all_values: list[np.ndarray] = []

    for filepath in list_vtp_files(case_dir):
        try:
            velocity = read_point_array(filepath, velocity_array)
        except KeyError:
            continue
        if velocity.ndim != 2 or velocity.shape[1] <= component_index:
            continue

        component = velocity[:, component_index]
        if component.size > 0:
            all_values.append(component)

    if not all_values:
        return None

    combined = np.concatenate(all_values)
    return CaseStatistics(case=case_dir.name, mean=float(np.mean(combined)), std=float(np.std(combined)), count=combined.size)


def compute_sweep_statistics(
    input_dir: Path,
    cases: Iterable[str],
    variable: Literal["contact_force", "velocity"],
    normalization: float = 1.0,
    outlier_strategy: OutlierStrategy = "none",
    value_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Compute per-case statistics for every case sub-folder in a parameter sweep."""
    rows: list[CaseStatistics] = []

    for case in cases:
        case_dir = input_dir / case
        if not case_dir.is_dir():
            print(f"[{case}] folder not found: {case_dir}")
            continue

        if variable == "contact_force":
            stats = compute_case_contact_force_statistics(
                case_dir,
                normalization=normalization,
                outlier_strategy=outlier_strategy,
                value_range=value_range,
            )
        elif variable == "velocity":
            stats = compute_case_velocity_statistics(case_dir)
        else:
            raise ValueError(f"Unknown variable: {variable}")

        if stats is None:
            print(f"[{case}] no valid data found.")
            continue

        rows.append(stats)
        print(f"[{stats.case}] mean={stats.mean:.4e} std={stats.std:.4e} count={stats.count}")

    return pd.DataFrame([r.__dict__ for r in rows])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Parent directory containing one sub-folder per case.")
    parser.add_argument("--cases", nargs="+", required=True, help="Case sub-folder names, in the desired output order.")
    parser.add_argument("--variable", choices=["contact_force", "velocity"], default="contact_force")
    parser.add_argument("--normalization", type=float, default=1.0, help="Reference value used to non-dimensionalize the contact force.")
    parser.add_argument("--outlier-strategy", choices=["none", "clip", "exclude"], default="none")
    parser.add_argument("--value-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path to write the summary table to.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    value_range = tuple(args.value_range) if args.value_range else None

    df = compute_sweep_statistics(
        input_dir=args.input_dir,
        cases=args.cases,
        variable=args.variable,
        normalization=args.normalization,
        outlier_strategy=args.outlier_strategy,
        value_range=value_range,
    )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
