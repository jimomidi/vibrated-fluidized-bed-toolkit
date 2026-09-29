"""Geometric measurement of wave/bubble structures from experimental video frames.

Each frame of a vibrated-bed experiment (saved as a grayscale image) is
thresholded and segmented into connected components ("blobs") using Otsu
thresholding followed by a morphological opening. For every blob that does
not touch the image border, its bounding-box height, width, height/width
ratio, equivalent circular diameter, and (optionally) its horizontal center
offset from mid-domain are computed in physical units, using a linear
pixel-to-physical-length scale supplied by the caller. Annotated copies of
the input images are written alongside the measurements for visual
verification.

Expected input layout
----------------------
    <input_dir>/*.png|*.jpg|*.jpeg   (one file per frame, sorted by filename)

Usage
-----
    python -m analysis.blob_image_measurement \\
        --input-dir data/frames/case_a \\
        --output-dir results/case_a \\
        --domain-width 1.0 --domain-height 0.5 \\
        --compute-center-offset
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class BlobMeasurements:
    """Per-frame lists of blob measurements (one entry per detected blob)."""

    height: list[float] = field(default_factory=list)
    width: list[float] = field(default_factory=list)
    height_width_ratio: list[float] = field(default_factory=list)
    diameter: list[float] = field(default_factory=list)
    center_x: list[float] = field(default_factory=list)
    center_x_transformed: list[float] = field(default_factory=list)


def list_frame_files(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def segment_blobs(image_gray: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Otsu-threshold and morphologically open a grayscale frame into labeled blobs."""
    _, thresholded = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.connectedComponentsWithStats(mask)


def measure_frame(
    image_path: Path,
    domain_width: float,
    domain_height: float,
    compute_center_offset: bool,
    center_offset_scale: float = 1.0,
    annotated_output_path: Path | None = None,
) -> BlobMeasurements:
    """Detect and measure all interior blobs in a single frame.

    ``domain_width``/``domain_height`` define the extent represented by the
    full image (in whatever length unit the caller works in -- e.g. a
    normalized bed width of 1.0), used to convert blob pixel measurements
    (H, V, H/V, D) to that unit. Blobs touching the image border are
    discarded (they are assumed to be cropped and not fully visible).

    ``center_offset_scale`` is a separate, independent physical width (e.g.
    the real bed width in meters) used only to re-express each blob's
    horizontal center as a signed offset from mid-domain, in that physical
    unit: ``center_x_transformed = center_x_fraction * center_offset_scale
    - center_offset_scale / 2``. It is independent of ``domain_width``
    because the two original measurement scripts this is based on used a
    normalized domain for H/V/D and a real physical width for the center
    offset.
    """
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    measurements = BlobMeasurements()
    if image_gray is None:
        return measurements

    height_px, width_px = image_gray.shape
    scale_x = domain_width / width_px
    scale_y = domain_height / height_px

    num_labels, _, stats, centroids = segment_blobs(image_gray)
    annotated = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    for label in range(1, num_labels):
        x, y, w, h, area_px = stats[label]
        touches_border = x == 0 or y == 0 or (x + w) == width_px or (y + h) == height_px
        if touches_border:
            continue

        width_phys = w * scale_x
        height_phys = h * scale_y
        area_phys = area_px * scale_x * scale_y
        diameter = math.sqrt(4 * area_phys / math.pi)

        measurements.height.append(height_phys)
        measurements.width.append(width_phys)
        measurements.height_width_ratio.append(height_phys / width_phys if width_phys != 0 else float("nan"))
        measurements.diameter.append(diameter)

        label_text = f"W={width_phys:.3f}, H={height_phys:.3f}, D={diameter:.3f}"

        if compute_center_offset:
            center_x_px, center_y_px = centroids[label]
            center_x_fraction = center_x_px / width_px
            center_x_scaled = center_x_fraction * domain_width
            center_x_transformed = center_x_fraction * center_offset_scale - center_offset_scale / 2
            measurements.center_x.append(center_x_scaled)
            measurements.center_x_transformed.append(center_x_transformed)
            label_text += f", CX={center_x_scaled:.3f}, CX'={center_x_transformed:.3f}"
            text_anchor = (int(center_x_px) - 70, int(center_y_px))
        else:
            cx, cy = centroids[label]
            text_anchor = (int(cx) - 50, int(cy))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, label_text, text_anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    if annotated_output_path is not None:
        cv2.imwrite(str(annotated_output_path), annotated)

    return measurements


def process_frame_folder(
    input_dir: Path,
    output_dir: Path,
    domain_width: float,
    domain_height: float,
    compute_center_offset: bool = False,
    center_offset_scale: float = 1.0,
) -> None:
    """Measure every frame in ``input_dir`` and write CSV summaries to ``output_dir``.

    Writes two files:

    - ``blob_measurements.csv``: one block of rows per frame, columns
      ``Frame, H, V, H/V, D``.
    - ``blob_center_offsets.csv`` (only if ``compute_center_offset``):
      per-blob center position and offset, plus a per-frame mean/std of
      the (absolute) offset.

    Annotated frames are written to ``output_dir/measured/``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "measured"
    annotated_dir.mkdir(exist_ok=True)

    frame_files = list_frame_files(input_dir)
    all_measurements: list[BlobMeasurements] = []

    for frame_path in frame_files:
        annotated_path = annotated_dir / f"{frame_path.stem}_measured.png"
        all_measurements.append(
            measure_frame(
                frame_path,
                domain_width,
                domain_height,
                compute_center_offset,
                center_offset_scale=center_offset_scale,
                annotated_output_path=annotated_path,
            )
        )

    _write_main_measurements_csv(output_dir / "blob_measurements.csv", all_measurements)
    if compute_center_offset:
        _write_center_offset_csv(output_dir / "blob_center_offsets.csv", all_measurements)

    print(f"Processed {len(frame_files)} frames. Results written to {output_dir}")


def _write_main_measurements_csv(csv_path: Path, all_measurements: list[BlobMeasurements]) -> None:
    rows: list[list[object]] = []
    for frame_index, measurements in enumerate(all_measurements, start=1):
        n_blobs = len(measurements.height)
        for blob_index in range(n_blobs):
            rows.append(
                [
                    f"Frame {frame_index}" if blob_index == 0 else "",
                    measurements.height[blob_index],
                    measurements.width[blob_index],
                    measurements.height_width_ratio[blob_index],
                    measurements.diameter[blob_index],
                ]
            )
        rows.append(["", "", "", "", ""])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "H", "V", "H/V", "D"])
        writer.writerows(rows)


def _write_center_offset_csv(csv_path: Path, all_measurements: list[BlobMeasurements]) -> None:
    rows: list[list[object]] = []
    for frame_index, measurements in enumerate(all_measurements, start=1):
        frame_label = f"Frame {frame_index}"
        for blob_index, (cx, cx_transformed) in enumerate(
            zip(measurements.center_x, measurements.center_x_transformed), start=1
        ):
            rows.append([frame_label, blob_index, cx, cx_transformed])

        if measurements.center_x_transformed:
            mean_val = float(np.mean(measurements.center_x_transformed))
            std_val = float(np.std(measurements.center_x_transformed))
            rows.append([frame_label, "Mean", "", mean_val])
            rows.append([frame_label, "Std", "", std_val])
            print(f"{frame_label}: mean(CX')={mean_val:.4f}, std(CX')={std_val:.4f}")

        rows.append(["", "", "", ""])

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Blob #", "Center_X (scaled)", "Center_X' (transformed)"])
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-width", type=float, required=True, help="Width unit represented by the full image (used for H/W/D).")
    parser.add_argument("--domain-height", type=float, required=True, help="Height unit represented by the full image (used for H/W/D).")
    parser.add_argument("--compute-center-offset", action="store_true")
    parser.add_argument(
        "--center-offset-scale",
        type=float,
        default=1.0,
        help="Physical bed width used only for the center-offset measurement (independent of --domain-width).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_frame_folder(
        args.input_dir,
        args.output_dir,
        domain_width=args.domain_width,
        domain_height=args.domain_height,
        compute_center_offset=args.compute_center_offset,
        center_offset_scale=args.center_offset_scale,
    )


if __name__ == "__main__":
    main()
