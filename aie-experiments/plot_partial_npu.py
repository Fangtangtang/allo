#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot partial-NPU bf16 GEMM performance envelopes."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import io
import itertools
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import gemm_partial_npu as experiment  # pylint: disable=wrong-import-position
import plot as base_plot  # pylint: disable=wrong-import-position


DEFAULT_SUMMARY = experiment.DEFAULT_OUTPUT_DIR / "summary.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots"


def default_summary(device: str = experiment.base.DEFAULT_DEVICE) -> Path:
    """Return the device-specific partial-GEMM summary CSV."""
    return experiment.default_output_dir(device) / "summary.csv"


def default_output_dir(device: str = experiment.base.DEFAULT_DEVICE) -> Path:
    """Return the shared device-specific plot directory."""
    return base_plot.default_output_dir(device)


SERIES_ORDER = experiment.VARIANT_ORDER
SERIES_LABELS = experiment.VARIANT_LABELS
SERIES_COLORS = {
    "manual": "#008cd8",
    "compiled": "#ef476f",
    "compiled-full-io": "#06a77d",
}
SERIES_LINESTYLES = {
    "manual": "-",
    "compiled": "-",
    "compiled-full-io": "--",
}
SERIES_MARKERS = {
    "manual": "o",
    "compiled": ".",
    "compiled-full-io": "x",
}
SUMMARY_FIELDS = {
    "variant",
    "plot_series",
    "flow",
    "status",
    "validation",
    "timed_validation_failure",
    "dtype",
    "M",
    "N",
    "K",
    "compute_columns",
    "mapping_columns",
    "mapping_rows",
    "device_columns",
    "filtered_gflops",
    "filtered_min_us",
    "filtered_max_us",
}


class PlotError(RuntimeError):
    """An invalid or incomplete partial-NPU plotting input."""


@dataclass(frozen=True)
class ResultPoint:
    """One logical partial-NPU plot point."""

    series: str
    compute_columns: int
    M: int
    N: int
    K: int
    tops: float
    lower_tops: float
    upper_tops: float

    @property
    def intensity(self) -> float:
        """Return bf16 arithmetic intensity in OPs/byte."""
        return base_plot.arithmetic_intensity(self.M, self.N, self.K, experiment.DTYPE)


def expected_actual_keys(
    columns: int,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> set[tuple[str, int, int, int, int]]:
    """Return expected physical summary keys for one compute width."""
    return set(
        itertools.product(
            experiment.physical_variants(columns, experiment.VARIANT_ORDER, device),
            (columns,),
            base_plot.DEFAULT_SIZES,
            experiment.default_matrix_ns(device),
            base_plot.DEFAULT_SIZES,
        )
    )


def expected_series(
    variant: str,
    columns: int,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> tuple[str, ...]:
    """Return logical series represented by one physical summary row."""
    if (
        variant == "compiled"
        and columns == experiment.base.device_config(device).max_columns
    ):
        return "compiled", "compiled-full-io"
    return (variant,)


def expected_device_columns(
    variant: str,
    columns: int,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> int:
    """Return the required device width for one physical plot row."""
    return experiment.device_columns_for(variant, columns, device)


def expected_mapping_shape(variant: str, columns: int) -> tuple[int, int]:
    """Return the required (rows, columns) mapping topology."""
    if variant == "compiled" and columns == 1:
        return 2, 2
    return 4, columns


def format_key(key: tuple[str, int, int, int, int]) -> str:
    """Format one physical configuration for diagnostics."""
    variant, columns, M, N, K = key
    return f"{variant}/{columns}x4/M{M}/N{N}/K{K}"


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    """Read and minimally validate a partial-NPU summary CSV."""
    try:
        source = summary_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise PlotError(f"Cannot open summary CSV {summary_path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        missing_fields = sorted(SUMMARY_FIELDS - fields)
        if missing_fields:
            raise PlotError(
                "Summary CSV is missing required column(s): "
                + ", ".join(missing_fields)
            )
        return list(reader)


def parse_plot_series(value: str | None) -> tuple[str, ...]:
    """Parse the semicolon-separated logical-series field."""
    return tuple(item for item in (value or "").split(";") if item)


def parse_positive_metric(row: dict[str, str], field: str, issues: list[str]) -> float:
    """Parse one required finite, positive metric and append diagnostics."""
    try:
        value = float(row[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError
        return value
    except (KeyError, TypeError, ValueError):
        issues.append(f"{field}={row.get(field)!r}")
        return math.nan


def provenance_issues(
    row: dict[str, str],
    variant: str,
    columns: int,
    device: str,
) -> list[str]:
    """Validate device provenance, accepting legacy XDNA1 blanks."""
    config = experiment.base.device_config(device)
    flow = "mlir-aie" if variant == "manual" else "allo"
    physical_columns = expected_device_columns(variant, columns, device)
    expected = {
        "device": device,
        "target": config.target,
        "backend_target": config.backend_target(flow, physical_columns),
        "npu2": config.npu2,
    }
    issues = []
    for field, expected_value in expected.items():
        actual = str(row.get(field, "") or "").strip()
        if not actual and device == experiment.base.DEFAULT_DEVICE:
            continue
        if actual != expected_value:
            issues.append(f"{field}={row.get(field)!r}")
    return issues


def validate_summary(
    rows: Sequence[dict[str, str]],
    columns: int,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> list[ResultPoint]:
    """Validate a complete compute-width sweep and expand logical series."""
    if columns not in experiment.default_columns(device):
        raise ValueError(f"Unsupported compute columns for {device}: {columns}")
    expected = expected_actual_keys(columns, device)
    seen: dict[tuple[str, int, int, int, int], list[int]] = {}
    parsed = []
    issues = []

    for row_number, row in enumerate(rows, start=2):
        try:
            row_columns = int(row["compute_columns"])
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"row {row_number}: malformed compute columns ({exc})")
            continue
        if row_columns != columns:
            continue
        try:
            key = (
                row["variant"],
                row_columns,
                int(row["M"]),
                int(row["N"]),
                int(row["K"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"row {row_number}: malformed configuration ({exc})")
            continue

        seen.setdefault(key, []).append(row_number)
        name = format_key(key)
        if key not in expected:
            issues.append(f"{name}: unexpected configuration")
            continue

        row_issues = provenance_issues(row, key[0], columns, device)
        if row.get("dtype") != experiment.DTYPE:
            row_issues.append(f"dtype={row.get('dtype')!r}")
        variant = key[0]
        expected_flow = "mlir-aie" if variant == "manual" else "allo"
        if row.get("flow") != expected_flow:
            row_issues.append(f"flow={row.get('flow')!r}")
        try:
            device_columns = int(row["device_columns"])
        except (KeyError, TypeError, ValueError):
            device_columns = -1
            row_issues.append(f"device_columns={row.get('device_columns')!r}")
        if device_columns != expected_device_columns(variant, columns, device):
            row_issues.append(f"device_columns={device_columns!r}")
        expected_rows, expected_columns = expected_mapping_shape(variant, columns)
        for field, expected_value in (
            ("mapping_rows", expected_rows),
            ("mapping_columns", expected_columns),
        ):
            try:
                actual_value = int(row[field])
            except (KeyError, TypeError, ValueError):
                row_issues.append(f"{field}={row.get(field)!r}")
                continue
            if actual_value != expected_value:
                row_issues.append(f"{field}={actual_value!r}")

        plot_series = parse_plot_series(row.get("plot_series"))
        if plot_series != expected_series(variant, columns, device):
            row_issues.append(f"plot_series={row.get('plot_series')!r}")

        try:
            timed_failure = base_plot.parse_timed_validation_failure(
                row.get("timed_validation_failure")
            )
        except ValueError:
            timed_failure = False
            row_issues.append(
                "timed_validation_failure=" f"{row.get('timed_validation_failure')!r}"
            )
        if timed_failure:
            if variant == "manual":
                row_issues.append("timed validation failure is invalid for manual")
            if row.get("status") != "failed":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "failed":
                row_issues.append(f"validation={row.get('validation')!r}")
        else:
            if row.get("status") != "success":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "passed":
                row_issues.append(f"validation={row.get('validation')!r}")

        filtered_gflops = parse_positive_metric(row, "filtered_gflops", row_issues)
        filtered_min_us = parse_positive_metric(row, "filtered_min_us", row_issues)
        filtered_max_us = parse_positive_metric(row, "filtered_max_us", row_issues)
        if (
            math.isfinite(filtered_min_us)
            and math.isfinite(filtered_max_us)
            and filtered_min_us > filtered_max_us
        ):
            row_issues.append("filtered_min_us exceeds filtered_max_us")

        if row_issues:
            issues.append(f"{name}: " + ", ".join(row_issues))
            continue
        for series in plot_series:
            parsed.append(
                ResultPoint(
                    series,
                    columns,
                    key[2],
                    key[3],
                    key[4],
                    filtered_gflops / 1000.0,
                    base_plot.tops_for_time(key[2], key[3], key[4], filtered_max_us),
                    base_plot.tops_for_time(key[2], key[3], key[4], filtered_min_us),
                )
            )

    for key, row_numbers in seen.items():
        if len(row_numbers) > 1:
            rows_text = ", ".join(str(value) for value in row_numbers)
            issues.append(f"{format_key(key)}: duplicate rows {rows_text}")
    for key in sorted(expected - seen.keys()):
        issues.append(f"{format_key(key)}: missing")

    expected_logical = (
        len(SERIES_ORDER)
        * len(base_plot.DEFAULT_SIZES)
        * len(experiment.default_matrix_ns(device))
        * len(base_plot.DEFAULT_SIZES)
    )
    if issues:
        details = "\n".join(f"  - {issue}" for issue in sorted(issues))
        raise PlotError(
            f"Summary CSV does not contain a complete, valid {columns}x4 sweep:\n"
            f"{details}"
        )
    if len(parsed) != expected_logical:
        raise PlotError(
            f"Expected {expected_logical} logical results for {columns}x4, "
            f"parsed {len(parsed)}"
        )
    return parsed


def performance_envelope(
    points: Iterable[ResultPoint], series: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return average, lower, and upper best envelopes by intensity."""
    maxima: dict[float, tuple[float, float, float]] = {}
    for point in points:
        if point.series != series:
            continue
        previous = maxima.get(point.intensity, (0.0, 0.0, 0.0))
        maxima[point.intensity] = (
            max(previous[0], point.tops),
            max(previous[1], point.lower_tops),
            max(previous[2], point.upper_tops),
        )
    if not maxima:
        raise PlotError(f"No points available for {series}")
    ordered = sorted(maxima.items())
    return (
        [item[0] for item in ordered],
        [item[1][0] for item in ordered],
        [item[1][1] for item in ordered],
        [item[1][2] for item in ordered],
    )


def create_figure(points: Sequence[ResultPoint], columns: int):
    """Create one partial-NPU performance-envelope figure."""
    figure, axis = plt.subplots(figsize=(5.0, 4.0))
    for series in SERIES_ORDER:
        intensities, tops, lower_tops, upper_tops = performance_envelope(points, series)
        axis.fill_between(
            intensities,
            lower_tops,
            upper_tops,
            facecolor=SERIES_COLORS[series],
            alpha=0.2,
        )
        axis.plot(
            intensities,
            tops,
            color=SERIES_COLORS[series],
            label=SERIES_LABELS[series],
            linestyle=SERIES_LINESTYLES[series],
            linewidth=2,
            marker=SERIES_MARKERS[series],
            markersize=6,
        )
    axis.set_title(f"bf16 — {columns}×4 compute tiles")
    axis.set_xlabel("Arithmetic Intensity (OPs/byte)")
    axis.set_ylabel("Performance (TOP/s)")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_axisbelow(True)
    axis.grid(True, color="lightgray", linestyle="--", alpha=0.7)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def render_png(points: Sequence[ResultPoint], columns: int) -> bytes:
    """Render one compute-width plot to PNG bytes."""
    figure = create_figure(points, columns)
    output = io.BytesIO()
    try:
        figure.savefig(output, format="png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output.getvalue()


def generate_plots(
    summary_path: Path,
    output_dir: Path,
    columns: Sequence[int] | None = None,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> tuple[list[Path], dict[int, str]]:
    """Write plots for complete compute widths and report skipped widths."""
    rows = read_summary(summary_path)
    selected_columns = columns or experiment.default_columns(device)
    rendered = {}
    skipped = {}
    for compute_columns in dict.fromkeys(selected_columns):
        try:
            points = validate_summary(rows, int(compute_columns), device)
        except (PlotError, ValueError) as exc:
            skipped[int(compute_columns)] = str(exc)
            continue
        rendered[int(compute_columns)] = render_png(points, int(compute_columns))
    if not rendered:
        details = "\n".join(
            f"{compute_columns}x4: {message}"
            for compute_columns, message in skipped.items()
        )
        raise PlotError(f"No compute-width configuration is complete:\n{details}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for compute_columns, image in rendered.items():
        path = output_dir / f"gemm_bf16_{compute_columns}x4.png"
        base_plot.atomic_write_bytes(path, image)
        paths.append(path)
    return paths, skipped


def build_parser() -> argparse.ArgumentParser:
    """Build the partial-NPU plotting parser."""
    parser = argparse.ArgumentParser(
        description="Plot partial-NPU bf16 GEMM performance envelopes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device",
        choices=experiment.base.DEVICE_CHOICES,
        default=experiment.base.DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary CSV (device-specific when omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="plot directory (device-specific when omitted)",
    )
    parser.add_argument(
        "--columns",
        type=int,
        choices=experiment.ALL_COLUMNS,
        nargs="+",
        default=None,
        help="compute widths (device-specific when omitted)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the partial-NPU plotting command-line interface."""
    args = build_parser().parse_args(argv)
    summary = args.summary or default_summary(args.device)
    output_dir = args.output_dir or default_output_dir(args.device)
    try:
        paths, skipped = generate_plots(summary, output_dir, args.columns, args.device)
    except PlotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for columns, message in skipped.items():
        print(f"warning: skipping {columns}x4: {message}", file=sys.stderr)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
