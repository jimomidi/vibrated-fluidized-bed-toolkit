"""Tracer-based particle mixing quantification for vibrated granular systems.

This module implements a two-step workflow for quantifying horizontal and
vertical particle mixing over time from a time series of DEM particle
point-cloud snapshots (``.vtp``):

1. **Tracer labeling** (:func:`label_tracer_particles`): using the *first*
   snapshot of a run, particles are split into two independent binary
   tracer groups based on their initial position -- a horizontal group
   (``h_tracer``) formed from alternating quarters of the domain width, and
   a vertical group (``v_tracer``) formed the same way along the domain
   height. Each particle keeps its assigned tracer values for the rest of
   the run (matched across snapshots by ``Particle_ID``), producing a new
   series of ``.vtp`` files with the two extra point-data arrays.

2. **Mixing index over time** (:func:`compute_mixing_index_timeseries`):
   for each labeled snapshot, particles are binned into a 3-D grid and the
   local tracer concentration in each occupied cell is used to compute the
   Ensemble Segregation Index (ESI), a normalized measure of how far the
   particle distribution is from perfectly mixed (ESI -> 0) vs. fully
   segregated (ESI -> 1).

Expected input
--------------
Each ``.vtp`` file must contain a ``Particle_ID`` point-data array (for
tracer step 1) and, after labeling, the ``h_tracer`` / ``v_tracer`` arrays
(for step 2). Point coordinates are used directly to derive tracer groups
and spatial bins.

Usage
-----
    python -m analysis.tracer_mixing_index label \\
        --input-dir data/raw_snapshots/case_a \\
        --output-dir data/labeled_snapshots/case_a

    python -m analysis.tracer_mixing_index mixing-index \\
        --input-dir data/labeled_snapshots/case_a \\
        --dt 0.02 --segment-size 0.002 \\
        --output-csv results/mixing_index_case_a.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def _natural_sort_key(path: Path) -> int:
    """Sort ``.vtp`` files by the numeric index embedded in their filename."""
    digits = "".join(filter(str.isdigit, path.stem))
    return int(digits) if digits else 0


def list_vtp_files_sorted(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.vtp"), key=_natural_sort_key)


def assign_quadrant_tracers(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split particles into alternating quarters of the domain, independently in x and y.

    Returns two integer arrays (``h_tracer``, ``v_tracer``) with values in
    ``{1, 2}``. Along each axis the domain is split into four equal
    quarters, and the two outer quarters (1st and 3rd) are grouped
    together (label 1) versus the two middle quarters (2nd and 4th,
    label 2), producing a checkerboard-like initial tracer pattern that is
    then tracked over time.
    """
    x_coords, y_coords = positions[:, 0], positions[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    x1, x2, x3 = (x_min + f * (x_max - x_min) for f in (0.25, 0.5, 0.75))
    y1, y2, y3 = (y_min + f * (y_max - y_min) for f in (0.25, 0.5, 0.75))

    h_tracer = np.where((x_coords < x1) | ((x_coords >= x2) & (x_coords < x3)), 1, 2)
    v_tracer = np.where((y_coords < y1) | ((y_coords >= y2) & (y_coords < y3)), 1, 2)
    return h_tracer, v_tracer


def label_tracer_particles(input_dir: Path, output_dir: Path) -> None:
    """Label every snapshot in ``input_dir`` with persistent tracer groups.

    The tracer assignment is derived once, from the first snapshot's
    particle positions, and then propagated to every later snapshot by
    matching ``Particle_ID``. Particles absent from the initial mapping
    (e.g. particles that enter later) are labeled ``-1``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer_by_id: dict[int, tuple[int, int]] = {}

    files = list_vtp_files_sorted(input_dir)
    print(f"Found {len(files)} VTP files in {input_dir}")

    for index, filepath in enumerate(files):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(filepath))
        reader.Update()
        polydata = reader.GetOutput()

        points = polydata.GetPoints()
        ids_array = polydata.GetPointData().GetArray("Particle_ID")
        if points is None or ids_array is None:
            print(f"Skipping {filepath.name}: missing points or Particle_ID.")
            continue

        num_points = points.GetNumberOfPoints()
        particle_ids = vtk_to_numpy(ids_array).astype(int)

        if index == 0:
            positions = vtk_to_numpy(points.GetData())
            h_values, v_values = assign_quadrant_tracers(positions)
            tracer_by_id = dict(zip(particle_ids.tolist(), zip(h_values.tolist(), v_values.tolist())))
            print(f"Tracer mapping created from {filepath.name}.")

        h_tracer = np.full(num_points, -1, dtype=np.int32)
        v_tracer = np.full(num_points, -1, dtype=np.int32)
        for i, pid in enumerate(particle_ids):
            if pid in tracer_by_id:
                h_tracer[i], v_tracer[i] = tracer_by_id[pid]

        for name, values in (("h_tracer", h_tracer), ("v_tracer", v_tracer)):
            vtk_array = numpy_to_vtk(values, deep=True)
            vtk_array.SetName(name)
            polydata.GetPointData().AddArray(vtk_array)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(output_dir / filepath.name))
        writer.SetInputData(polydata)
        writer.Write()
        print(f"Saved labeled file: {filepath.name}")


def compute_ensemble_segregation_index(concentrations: np.ndarray, weights: np.ndarray) -> float:
    """Compute the Ensemble Segregation Index (ESI) of a local-concentration field.

    ESI = std(c) / sqrt(mean(c) * (1 - mean(c))), weighted by local particle
    counts. Returns 0 for an empty, fully mixed (mean 0 or 1), or
    zero-weight field.
    """
    if concentrations.size == 0 or np.sum(weights) == 0:
        return 0.0
    mean_c = np.average(concentrations, weights=weights)
    if mean_c in (0, 1):
        return 0.0
    std_c = np.sqrt(np.average((concentrations - mean_c) ** 2, weights=weights))
    return float(std_c / np.sqrt(mean_c * (1 - mean_c)))


def compute_snapshot_mixing_indices(filepath: Path, segment_size: float) -> tuple[float, float]:
    """Compute the horizontal and vertical ESI for one labeled snapshot."""
    mesh = pv.read(str(filepath))
    points = mesh.points
    point_data = mesh.point_data

    if "h_tracer" not in point_data or "v_tracer" not in point_data:
        print(f"{filepath.name} missing tracer data.")
        return 0.0, 0.0

    x, y, z = points.T
    if x.size == 0:
        return 0.0, 0.0

    bounds_min = np.array([x.min(), y.min(), z.min()])
    bounds_max = np.array([x.max(), y.max(), z.max()])
    grid_shape = np.maximum(np.ceil((bounds_max - bounds_min) / segment_size).astype(int), 1)

    h_concentration = np.zeros(grid_shape)
    v_concentration = np.zeros(grid_shape)
    counts = np.zeros(grid_shape)

    cell_index = np.minimum(
        ((points - bounds_min) / segment_size).astype(int), grid_shape - 1
    )
    tracer_h = point_data["h_tracer"]
    tracer_v = point_data["v_tracer"]

    for (xi, yi, zi), h_val, v_val in zip(cell_index, tracer_h, tracer_v):
        counts[xi, yi, zi] += 1
        if h_val == 1:
            h_concentration[xi, yi, zi] += 1
        if v_val == 1:
            v_concentration[xi, yi, zi] += 1

    with np.errstate(invalid="ignore"):
        h_concentration = np.divide(h_concentration, counts, out=np.zeros_like(h_concentration), where=counts > 0)
        v_concentration = np.divide(v_concentration, counts, out=np.zeros_like(v_concentration), where=counts > 0)

    weights = counts.flatten()
    h_esi = compute_ensemble_segregation_index(h_concentration.flatten(), weights)
    v_esi = compute_ensemble_segregation_index(v_concentration.flatten(), weights)
    return h_esi, v_esi


def compute_mixing_index_timeseries(input_dir: Path, dt: float, segment_size: float) -> pd.DataFrame:
    """Compute the horizontal/vertical ESI time series for a labeled snapshot series."""
    files = list_vtp_files_sorted(input_dir)

    records = []
    current_time = 0.0
    for filepath in files:
        print(f"Processing {filepath.name} ...")
        h_esi, v_esi = compute_snapshot_mixing_indices(filepath, segment_size)
        records.append({"time_s": current_time, "HorizontalESI": h_esi, "VerticalESI": v_esi})
        current_time += dt

    return pd.DataFrame.from_records(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("label", help="Assign persistent tracer groups to a snapshot series.")
    label_parser.add_argument("--input-dir", type=Path, required=True)
    label_parser.add_argument("--output-dir", type=Path, required=True)

    mixing_parser = subparsers.add_parser("mixing-index", help="Compute the ESI mixing index over time.")
    mixing_parser.add_argument("--input-dir", type=Path, required=True, help="Directory of tracer-labeled snapshots.")
    mixing_parser.add_argument("--dt", type=float, required=True, help="Time interval between consecutive snapshots (s).")
    mixing_parser.add_argument("--segment-size", type=float, required=True, help="Spatial bin size used to estimate local concentration.")
    mixing_parser.add_argument("--output-csv", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "label":
        label_tracer_particles(args.input_dir, args.output_dir)
    elif args.command == "mixing-index":
        df = compute_mixing_index_timeseries(args.input_dir, dt=args.dt, segment_size=args.segment_size)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved mixing index time series to {args.output_csv}")


if __name__ == "__main__":
    main()
