# Vibrated Fluidized Bed Toolkit

Post-processing utilities for computational (CFD/DEM) and experimental
studies of vibrated gas-fluidized beds. The toolkit was developed to reduce
raw particle-resolved simulation output and high-speed video frames into the
per-case statistics, mixing metrics, and figures typically needed for this
kind of parameter-sweep study, and to remove the large amount of ad hoc,
one-off duplication that tends to accumulate in an exploratory research
codebase.

It is organized into two packages:

- **`analysis/`** — computation: reading simulation output and video frames,
  computing statistics, mixing indices, and correlations, and writing tidy
  CSV summaries.
- **`figures/`** — presentation: a reusable, parameterized plotting library
  that turns those CSV summaries into publication-style figures.

## What it does

| Module | Purpose |
|---|---|
| `analysis/contact_force_velocity_stats.py` | Aggregates per-particle contact-force magnitude and velocity statistics from a sweep of VTK PolyData (`.vtp`) snapshots, one sub-folder per parameter case. |
| `analysis/tracer_mixing_index.py` | Labels particles with persistent horizontal/vertical tracer groups from their initial position, then computes an Ensemble Segregation Index (ESI) mixing metric over time. |
| `analysis/void_fraction_correlation.py` | Computes the lag-correlation of the gas void fraction field between snapshots (on unstructured `.vtu` grids), with a multi-stage robust outlier filter, to characterize wave periodicity. |
| `analysis/blob_image_measurement.py` | Detects and measures wave/bubble structures (height, width, aspect ratio, equivalent diameter, and optional center offset) from grayscale experimental video frames using OpenCV. |
| `analysis/frame_statistics.py` | Aggregates per-frame blob measurements across many per-case CSVs into case-by-time summary tables and coefficient-of-variation ratios. |
| `figures/plotting.py` | Reusable Matplotlib/Seaborn functions: error-bar-by-case plots, dual-axis error bars, case-by-time heatmaps, joint scatter/marginal-KDE plots, jittered boxplots, multi-series time series, striped time-evolution plots, and grouped bar-with-error charts. |

## Typical workflow

1. Run your CFD/DEM simulation (this toolkit was built around
   [MFiX](https://mfix.netl.doe.gov/)-style two-fluid/DEM output, but any
   solver that exports VTK PolyData/UnstructuredGrid snapshots will work)
   and organize the output as one sub-folder per parameter case.
2. Use `contact_force_velocity_stats.py` and/or `void_fraction_correlation.py`
   to reduce the raw snapshots to per-case statistics.
3. If you need particle-mixing metrics, run `tracer_mixing_index.py label`
   once per case to add tracer arrays, then `tracer_mixing_index.py
   mixing-index` to get an ESI time series.
4. For experimental video, run `blob_image_measurement.py` per case to
   extract blob measurements, then `frame_statistics.py` to aggregate them
   across frames and cases.
5. Import functions from `figures/plotting.py` (or call it as a library) to
   turn the resulting CSVs into figures.

Each analysis module can also be used as a library — every CLI is a thin
wrapper around plain functions with type hints and docstrings, so you can
import and compose them directly in your own scripts or notebooks.

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

## Usage

Each module can be run as a script, e.g.:

```bash
python -m analysis.contact_force_velocity_stats \
    --input-dir data/contact_force/frequency_ratio_sweep \
    --cases 000 025 050 075 100 150 200 250 \
    --variable contact_force \
    --normalization 1.67e-7 \
    --output-csv results/contact_force_by_case.csv

python -m analysis.tracer_mixing_index label \
    --input-dir data/raw_snapshots/case_a \
    --output-dir data/labeled_snapshots/case_a

python -m analysis.tracer_mixing_index mixing-index \
    --input-dir data/labeled_snapshots/case_a \
    --dt 0.02 --segment-size 0.002 \
    --output-csv results/mixing_index_case_a.csv

python -m analysis.void_fraction_correlation \
    --root-dir data/void_fraction/vibration_strength_sweep \
    --delta-t 14 --variable EP_G --y-bounds 0.015 0.035 \
    --output-csv results/void_fraction_correlation.csv

python -m analysis.blob_image_measurement \
    --input-dir data/frames/case_a \
    --output-dir results/case_a \
    --domain-width 1.0 --domain-height 0.5 \
    --compute-center-offset --center-offset-scale 0.1

python -m analysis.frame_statistics case-time-matrix \
    --input-dir data/blob_measurements/frequency_ratio_sweep \
    --value-column "H/V" --num-frames 15 \
    --output-csv results/hv_by_case_and_frame.csv
```

Run any module with `--help` to see its full set of options, and see each
module's top-of-file docstring for the exact expected input layout and
data-array names.

Run `python -m figures.plotting` on its own to smoke-test the plotting
library — it renders every figure type once with synthetic, randomly
generated placeholder data purely to demonstrate the API. It does not
produce or represent any real result.

## Expected input formats

- **VTK PolyData (`.vtp`)** snapshots need a `Particle_ID` point array (for
  tracer labeling) and whichever point-data arrays you pass as
  `--variable`/component names (defaults assume a 3-component contact-force
  vector named `User_Defined_Var_1/2/3` and a 3-component `Velocity` array,
  as exported by some DEM/TFM solvers — override these via the function
  arguments if your solver names them differently).
- **VTK UnstructuredGrid (`.vtu`)** snapshots need the configured cell-data
  scalar field (default `EP_G`, i.e. gas void fraction).
- **Video frames** should be grayscale image files (`.png`/`.jpg`/`.jpeg`),
  one per time step, sortable by filename.
- **Measurement CSVs** consumed by `frame_statistics.py` should follow the
  column layout produced by `blob_image_measurement.py` (a `Frame` column
  plus one or more numeric measurement columns).

## Reproducibility and customization

- All input/output paths, sweep-case names, array names, and normalization
  constants are passed as function arguments or CLI flags — nothing is
  hard-coded to a particular machine or directory layout.
- Numerical thresholds that were tuned empirically in the original analysis
  (contact-force clipping ranges, the ESI spatial bin size, the robust
  correlation filter's MAD/Tukey/Hampel parameters) are exposed as
  parameters with the original defaults, documented in each function's
  docstring, rather than being silently fixed.
- Random behavior is limited to cosmetic scatter-plot jitter in
  `plot_boxplot_with_jitter` and the synthetic data generated in
  `figures/plotting.py`'s demo block; none of the analysis computations are
  stochastic.

## Known open questions from the original analysis

A couple of small inconsistencies were present in the exploratory code this
toolkit was distilled from and are worth resolving deliberately rather than
silently:

- Two variants of the outlier handling for the normalized contact force
  were used interchangeably in the original notebooks: clipping values to a
  fixed range vs. excluding them outright. Both are preserved here as an
  explicit `outlier_strategy` option (`"none"`, `"clip"`, `"exclude"`) in
  `contact_force_velocity_stats.py`; pick the one that matches your
  intended analysis, as they give different summary statistics.
- The center-offset statistic in `analysis/blob_image_measurement.py`
  reports the mean/std of the *signed* per-frame transformed offset; this
  is the version that was implemented unambiguously in the original
  analysis. A separate, narrower pair of scripts additionally computed a
  single mean/std across an *entire run* (all frames combined) rather than
  per frame, and disagreed with each other on sign convention (one took
  the absolute value first, one did not). That whole-run aggregate was not
  carried into this toolkit, since it is not the more complete/functional
  original version and is trivially recoverable from the per-frame CSV
  that `blob_image_measurement.py` already writes (average the
  `center_x_transformed` column, or its `abs()`, across all frames
  yourself if you need that summary).

## Notes from the refactoring audit

A follow-up audit checked every module's logic against the original
notebooks line by line and searched for any accidentally hard-coded
analytical data. No hard-coded results or fabricated data were found
anywhere in `analysis/` or `figures/`; every numeric constant in the code
is either a documented, user-overridable analysis parameter (e.g. the
robust-filter thresholds in `void_fraction_correlation.py`) or synthetic
demo data clearly marked as such (`figures/plotting.py`'s `__main__`
block). One undisclosed defensive fix, made silently during the original
port, was found and is documented here for transparency:

- `tracer_mixing_index.py`'s spatial binning
  (`compute_snapshot_mixing_indices`) floors the computed grid size to at
  least 1 cell per axis (`np.maximum(grid_shape, 1)`) before indexing. The
  original notebook code had no such floor and would raise an `IndexError`
  if a snapshot's spatial bounds were exactly constant on any axis (e.g. a
  perfectly flat coordinate in a pseudo-2D domain), since
  `ceil(0 / segment_size)` is `0` and a zero-length axis cannot be
  negative-indexed. For any dataset with non-zero extent on every axis
  (the normal case), `np.ceil` of a strictly positive value is already
  `>= 1`, so this floor never changes a computed mixing-index value -- it
  only prevents a crash on a degenerate edge case.

## License

MIT — see `LICENSE`.
