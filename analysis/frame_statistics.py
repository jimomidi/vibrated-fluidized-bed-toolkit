"""Aggregation of per-blob measurement CSVs into per-case, per-frame summary tables.

This module operates on the CSV output of :mod:`analysis.blob_image_measurement`
(or any CSV sharing the same shape) and provides three related aggregations:

1. :func:`aggregate_case_time_matrix` -- given one CSV per case, each with a
   ``Frame`` column and one or more numeric measurement columns, compute the
   mean of a chosen column for each of a fixed number of frames, producing a
   case x frame matrix (one row per input file / case).

2. :func:`compute_std_to_mean_ratio` -- given two matching case x frame
   matrices of means and standard deviations, compute the element-wise
   coefficient of variation (std / mean).

3. :func:`compute_column_statistics` -- generic per-column mean/std for a
   folder of CSV files, used for simple derived-quantity summaries (e.g. a
   per-frame diameter or center-distance table) that do not need the
   case x frame reshaping above.

Usage
-----
    python -m analysis.frame_statistics case-time-matrix \\
        --input-dir data/blob_measurements/frequency_ratio_sweep \\
        --value-column "H/V" --num-frames 15 \\
        --output-csv results/hv_by_case_and_frame.csv

    python -m analysis.frame_statistics column-stats \\
        --input-dir data/derived_measurements --output-dir results/stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        matches = [c for c in df.columns if c.strip().upper() == candidate.upper()]
        if matches:
            return matches[0]
    raise KeyError(f"None of the expected columns {candidates} found in {list(df.columns)}")


def aggregate_case_time_matrix(
    input_dir: Path,
    value_column: str,
    num_frames: int,
    include_std: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Build a (case x frame) matrix of the mean of ``value_column`` per frame.

    Every ``*.csv`` file in ``input_dir`` is treated as one case/row. Each
    file is expected to have a ``Frame`` column with values like
    ``"Frame 1"``, ``"Frame 2"``, ... and a column matching
    ``value_column`` (case-insensitive, whitespace-tolerant).
    """
    files = sorted(input_dir.glob("*.csv"))
    time_columns = [f"time {i}" for i in range(1, num_frames + 1)]

    mean_rows, std_rows = [], []
    for filepath in files:
        df = pd.read_csv(filepath)
        frame_column = df.columns[0]
        value_col = _find_column(df, [value_column])

        means, stds = [], []
        for frame_index in range(1, num_frames + 1):
            mask = df[frame_column].astype(str).str.strip() == f"Frame {frame_index}"
            values = pd.to_numeric(df.loc[mask, value_col], errors="coerce").dropna()
            means.append(values.mean() if not values.empty else float("nan"))
            stds.append(values.std(ddof=0) if not values.empty else float("nan"))

        mean_rows.append(means)
        std_rows.append(stds)

    mean_df = pd.DataFrame(mean_rows, columns=time_columns)
    mean_df.insert(0, "case", range(len(mean_rows)))

    if not include_std:
        return mean_df

    std_df = pd.DataFrame(std_rows, columns=time_columns)
    std_df.insert(0, "case", range(len(std_rows)))
    return mean_df, std_df


def compute_std_to_mean_ratio(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    """Element-wise std/mean (coefficient of variation), preserving the ``case`` column."""
    ratio_df = std_df.copy()
    for column in mean_df.columns:
        if column != "case":
            ratio_df[column] = std_df[column] / mean_df[column]
    return ratio_df


def compute_column_statistics(input_dir: Path, output_dir: Path) -> None:
    """Write per-column mean/std for every numeric column of every CSV in ``input_dir``.

    For each ``<name>.csv``, writes ``<name>_stats.csv`` with columns
    ``Frame, Mean, Std`` (one row per original column, treated as a
    labeled frame/measurement).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(filepath, sep=",", engine="python")
        df = df.dropna(how="all").apply(pd.to_numeric, errors="coerce")

        stats_df = pd.DataFrame({"Frame": df.mean().index, "Mean": df.mean().values, "Std": df.std().values})
        output_path = output_dir / f"{filepath.stem}_stats.csv"
        stats_df.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(stats_df)} rows)")


def compute_labeled_frame_statistics(input_dir: Path, output_dir: Path, value_column: str) -> None:
    """Group a long-format CSV (with a ``Frame`` text column) by frame number and summarize.

    Each input CSV is expected to contain a ``Frame`` column with values
    like ``"Frame 3"`` and a numeric column named ``value_column``; the
    output has one row per distinct frame number with its mean and std.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(filepath, sep="\t|,", engine="python")
        df = df.dropna(how="all")
        df.columns = df.columns.str.strip()

        if "Frame" not in df.columns:
            print(f"'Frame' column not found in {filepath.name}, skipping.")
            continue
        if value_column not in df.columns:
            print(f"'{value_column}' column not found in {filepath.name}, skipping.")
            continue

        df["Frame"] = df["Frame"].astype(str).str.strip()
        df = df[df["Frame"].str.contains("Frame", na=False)]
        df["Frame_Num"] = df["Frame"].str.extract(r"(\d+)").astype(int)
        df[value_column] = pd.to_numeric(df[value_column], errors="coerce")

        stats = (
            df.groupby("Frame_Num")[value_column]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"Frame_Num": "Frame", "mean": f"{value_column}_Mean", "std": f"{value_column}_Std"})
        )

        output_path = output_dir / f"{filepath.stem}_stats.csv"
        stats.to_csv(output_path, index=False)
        print(f"Wrote {output_path} ({len(stats)} frames)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matrix_parser = subparsers.add_parser("case-time-matrix")
    matrix_parser.add_argument("--input-dir", type=Path, required=True)
    matrix_parser.add_argument("--value-column", required=True)
    matrix_parser.add_argument("--num-frames", type=int, required=True)
    matrix_parser.add_argument("--include-std", action="store_true")
    matrix_parser.add_argument("--output-csv", type=Path, required=True)

    ratio_parser = subparsers.add_parser("std-mean-ratio")
    ratio_parser.add_argument("--mean-csv", type=Path, required=True)
    ratio_parser.add_argument("--std-csv", type=Path, required=True)
    ratio_parser.add_argument("--output-csv", type=Path, required=True)

    column_parser = subparsers.add_parser("column-stats")
    column_parser.add_argument("--input-dir", type=Path, required=True)
    column_parser.add_argument("--output-dir", type=Path, required=True)

    labeled_parser = subparsers.add_parser("labeled-frame-stats")
    labeled_parser.add_argument("--input-dir", type=Path, required=True)
    labeled_parser.add_argument("--output-dir", type=Path, required=True)
    labeled_parser.add_argument("--value-column", required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "case-time-matrix":
        result = aggregate_case_time_matrix(args.input_dir, args.value_column, args.num_frames, args.include_std)
        mean_df = result[0] if args.include_std else result
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        mean_df.to_csv(args.output_csv, index=False)
        print(f"Saved case-time matrix to {args.output_csv}")

    elif args.command == "std-mean-ratio":
        ratio_df = compute_std_to_mean_ratio(pd.read_csv(args.mean_csv), pd.read_csv(args.std_csv))
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        ratio_df.to_csv(args.output_csv, index=False)
        print(f"Saved std/mean ratio table to {args.output_csv}")

    elif args.command == "column-stats":
        compute_column_statistics(args.input_dir, args.output_dir)

    elif args.command == "labeled-frame-stats":
        compute_labeled_frame_statistics(args.input_dir, args.output_dir, args.value_column)


if __name__ == "__main__":
    main()
