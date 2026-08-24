#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot GEMM arithmetic-intensity performance envelopes."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import io
import itertools
import math
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY = EXPERIMENT_DIR / "results" / "gemm" / "summary.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots"
DEFAULT_SIZES = (256, 512, 1024, 2048)
DTYPE_ORDER = ("int16", "int8", "bf16")
DTYPE_BYTES = {"int16": 2, "int8": 1, "bf16": 2}
FLOW_ORDER = ("allo", "mlir-aie")
FLOW_LABELS = {"allo": "Allo", "mlir-aie": "MLIR-AIE"}
FLOW_COLORS = {"allo": "#ef476f", "mlir-aie": "#008cd8"}
SUMMARY_FIELDS = {
    "flow",
    "status",
    "validation",
    "timed_validation_failure",
    "dtype",
    "M",
    "N",
    "K",
    "filtered_gflops",
    "filtered_min_us",
    "filtered_max_us",
}


class PlotError(RuntimeError):
    """An invalid or incomplete plotting input."""


@dataclass(frozen=True)
class ResultPoint:
    """One validated GEMM summary result."""

    flow: str
    dtype: str
    M: int
    N: int
    K: int
    tops: float
    lower_tops: float
    upper_tops: float

    @property
    def intensity(self) -> float:
        """Return arithmetic intensity in OPs/byte."""
        return arithmetic_intensity(self.M, self.N, self.K, self.dtype)


def arithmetic_intensity(M: int, N: int, K: int, dtype: str) -> float:
    """Compute GEMM arithmetic intensity in OPs/byte."""
    try:
        dtype_bytes = DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported datatype: {dtype}") from exc
    if min(M, N, K) <= 0:
        raise ValueError("GEMM dimensions must be positive")
    operations = 2.0 * M * N * K
    transferred_bytes = dtype_bytes * (M * K + K * N + M * N)
    return operations / transferred_bytes


def tops_for_time(M: int, N: int, K: int, time_us: float) -> float:
    """Compute GEMM performance in TOP/s from a time in microseconds."""
    if not math.isfinite(time_us) or time_us <= 0:
        raise ValueError("GEMM execution time must be positive and finite")
    operations = 2.0 * M * N * K
    return operations / (time_us * 1.0e6)


def expected_keys(
    dtypes: Sequence[str] = DTYPE_ORDER,
) -> set[tuple[str, str, int, int, int]]:
    """Return the configurations in the selected default datatype sweeps."""
    return set(
        itertools.product(
            FLOW_ORDER,
            dtypes,
            DEFAULT_SIZES,
            DEFAULT_SIZES,
            DEFAULT_SIZES,
        )
    )


def format_key(key: tuple[str, str, int, int, int]) -> str:
    """Format one configuration for validation diagnostics."""
    flow, dtype, M, N, K = key
    return f"{flow}/{dtype}/M{M}/N{N}/K{K}"


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    """Read and minimally validate a summary CSV."""
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


def parse_timed_validation_failure(value: str | None) -> bool:
    """Parse the aggregate marker while accepting legacy successful blanks."""
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"", "false", "0"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def validate_summary(
    rows: Sequence[dict[str, str]], dtypes: Sequence[str] = DTYPE_ORDER
) -> list[ResultPoint]:
    """Validate selected datatype sweeps and return parsed result points."""
    unsupported = sorted(set(dtypes) - set(DTYPE_ORDER))
    if unsupported:
        raise ValueError(f"Unsupported datatype(s): {', '.join(unsupported)}")
    expected = expected_keys(dtypes)
    seen: dict[tuple[str, str, int, int, int], list[int]] = {}
    parsed: list[ResultPoint] = []
    issues = []

    for row_number, row in enumerate(rows, start=2):
        if row.get("dtype") not in dtypes:
            continue
        try:
            key = (
                row["flow"],
                row["dtype"],
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

        row_issues = []
        try:
            timed_validation_failure = parse_timed_validation_failure(
                row.get("timed_validation_failure")
            )
        except ValueError:
            timed_validation_failure = False
            row_issues.append(
                "timed_validation_failure=" f"{row.get('timed_validation_failure')!r}"
            )
        if timed_validation_failure:
            if key[:2] != ("allo", "bf16"):
                row_issues.append(
                    "timed validation failure is only valid for allo/bf16"
                )
            if row.get("status") != "failed":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "failed":
                row_issues.append(f"validation={row.get('validation')!r}")
        else:
            if row.get("status") != "success":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "passed":
                row_issues.append(f"validation={row.get('validation')!r}")
        try:
            filtered_gflops = float(row["filtered_gflops"])
            if not math.isfinite(filtered_gflops) or filtered_gflops <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            filtered_gflops = math.nan
            row_issues.append(f"filtered_gflops={row.get('filtered_gflops')!r}")
        try:
            filtered_min_us = float(row["filtered_min_us"])
            if not math.isfinite(filtered_min_us) or filtered_min_us <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            filtered_min_us = math.nan
            row_issues.append(f"filtered_min_us={row.get('filtered_min_us')!r}")
        try:
            filtered_max_us = float(row["filtered_max_us"])
            if not math.isfinite(filtered_max_us) or filtered_max_us <= 0:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            filtered_max_us = math.nan
            row_issues.append(f"filtered_max_us={row.get('filtered_max_us')!r}")
        if (
            math.isfinite(filtered_min_us)
            and math.isfinite(filtered_max_us)
            and filtered_min_us > filtered_max_us
        ):
            row_issues.append("filtered_min_us exceeds filtered_max_us")

        if row_issues:
            issues.append(f"{name}: " + ", ".join(row_issues))
            continue
        parsed.append(
            ResultPoint(
                flow=key[0],
                dtype=key[1],
                M=key[2],
                N=key[3],
                K=key[4],
                tops=filtered_gflops / 1000.0,
                lower_tops=tops_for_time(key[2], key[3], key[4], filtered_max_us),
                upper_tops=tops_for_time(key[2], key[3], key[4], filtered_min_us),
            )
        )

    for key, row_numbers in seen.items():
        if len(row_numbers) > 1:
            rows_text = ", ".join(str(value) for value in row_numbers)
            issues.append(f"{format_key(key)}: duplicate rows {rows_text}")
    for key in sorted(expected - seen.keys()):
        issues.append(f"{format_key(key)}: missing")

    if issues:
        details = "\n".join(f"  - {issue}" for issue in sorted(issues))
        dtype_text = ", ".join(dtypes)
        raise PlotError(
            f"Summary CSV does not contain a complete, valid {len(expected)}-case "
            f"sweep for {dtype_text}:\n{details}"
        )
    if len(parsed) != len(expected):
        raise PlotError(f"Expected {len(expected)} valid results, parsed {len(parsed)}")
    return parsed


def performance_envelope(
    points: Iterable[ResultPoint], flow: str, dtype: str
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return average, lower, and upper best envelopes by intensity."""
    maxima: dict[float, tuple[float, float, float]] = {}
    for point in points:
        if point.flow != flow or point.dtype != dtype:
            continue
        previous = maxima.get(point.intensity, (0.0, 0.0, 0.0))
        maxima[point.intensity] = (
            max(previous[0], point.tops),
            max(previous[1], point.lower_tops),
            max(previous[2], point.upper_tops),
        )
    if not maxima:
        raise PlotError(f"No points available for {flow}/{dtype}")
    ordered = sorted(maxima.items())
    return (
        [item[0] for item in ordered],
        [item[1][0] for item in ordered],
        [item[1][1] for item in ordered],
        [item[1][2] for item in ordered],
    )


def create_figure(points: Sequence[ResultPoint], dtype: str):
    """Create one datatype performance-envelope figure."""
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for flow in FLOW_ORDER:
        intensities, tops, lower_tops, upper_tops = performance_envelope(
            points, flow, dtype
        )
        axis.fill_between(
            intensities,
            lower_tops,
            upper_tops,
            facecolor=FLOW_COLORS[flow],
            alpha=0.3,
        )
        axis.plot(
            intensities,
            tops,
            color=FLOW_COLORS[flow],
            label=FLOW_LABELS[flow],
            linewidth=2,
            marker=".",
            markersize=9,
        )
    axis.set_title(dtype)
    axis.set_xlabel("Arithmetic Intensity (OPs/byte)")
    axis.set_ylabel("Performance (TOP/s)")
    axis.set_xlim(left=0)
    axis.set_ylim(bottom=0)
    axis.set_axisbelow(True)
    axis.grid(True, color="lightgray", linestyle="--", alpha=0.7)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def render_png(points: Sequence[ResultPoint], dtype: str) -> bytes:
    """Render one plot to PNG bytes without touching the output directory."""
    figure = create_figure(points, dtype)
    output = io.BytesIO()
    try:
        figure.savefig(output, format="png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output.getvalue()


def atomic_write_bytes(path: Path, value: bytes) -> None:
    """Write bytes without exposing a partially written plot."""
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            output.write(value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_plots(
    summary_path: Path, output_dir: Path
) -> tuple[list[Path], dict[str, str]]:
    """Write plots for complete datatype sweeps and report skipped datatypes."""
    rows = read_summary(summary_path)
    rendered = {}
    skipped = {}
    for dtype in DTYPE_ORDER:
        try:
            points = validate_summary(rows, (dtype,))
        except PlotError as exc:
            skipped[dtype] = str(exc)
            continue
        rendered[dtype] = render_png(points, dtype)
    if not rendered:
        details = "\n".join(f"{dtype}: {message}" for dtype, message in skipped.items())
        raise PlotError(f"No datatype has a complete, valid sweep:\n{details}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dtype, image in rendered.items():
        path = output_dir / f"gemm_{dtype}.png"
        atomic_write_bytes(path, image)
        paths.append(path)
    return paths, skipped


def build_parser() -> argparse.ArgumentParser:
    """Build the plotting command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot GEMM performance envelopes by arithmetic intensity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the plotting command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        paths, skipped = generate_plots(args.summary, args.output_dir)
    except PlotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for dtype, message in skipped.items():
        print(f"warning: skipping {dtype}: {message}", file=sys.stderr)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
