#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot stacked end-to-end attention timing breakdowns."""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import attention as experiment  # pylint: disable=wrong-import-position

DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots"
IMPLEMENTATION_COMPONENT_COLORS = {
    "baseline": {"npu": "#0072B2", "extra": "#9ECAE1"},
    "flash": {"npu": "#B2182B", "extra": "#F4A3B4"},
}
IMPLEMENTATION_LEGEND_LABELS = {"baseline": "Unfused", "flash": "Fused"}
INFEASIBLE_COLOR = "#d62728"
SUMMARY_FIELDS = {
    "device",
    "target",
    "backend_target",
    "npu2",
    "implementation",
    "status",
    "validation",
    "timed_validation_failure",
    "dtype",
    "seq_len",
    "head_dim",
    "q_chunk_size",
    "kv_chunk_size",
    "mapping_rows",
    "mapping_columns",
    "compute_slots",
    "kernel_count",
    "timing_scope",
    "attention_timing_version",
    "npu_timing_scope",
    "npu_aggregation",
    "filtered_mean_us",
    "filtered_min_us",
    "filtered_max_us",
    "filtered_npu_mean_us",
    "filtered_extra_mean_us",
}


class PlotError(RuntimeError):
    """An invalid or incomplete attention plotting input."""


@dataclass(frozen=True)
class ResultPoint:
    """One validated attention timing breakdown."""

    implementation: str
    seq_len: int
    e2e_mean_us: float | None
    e2e_min_us: float | None
    e2e_max_us: float | None
    npu_mean_us: float | None
    extra_mean_us: float | None
    infeasible: bool = False


def default_summary(device: str = experiment.base.DEFAULT_DEVICE) -> Path:
    """Return the device-specific attention summary CSV."""
    return experiment.default_output_dir(device) / "summary.csv"


def default_output_dir(device: str = experiment.base.DEFAULT_DEVICE) -> Path:
    """Return the shared device-specific plot directory."""
    if device == "xdna2":
        return DEFAULT_OUTPUT_DIR / "xdna2"
    experiment.base.device_config(device)
    return DEFAULT_OUTPUT_DIR


def expected_keys() -> set[tuple[str, int]]:
    """Return all configurations required for one complete device sweep."""
    return {
        (implementation, seq_len)
        for implementation in experiment.IMPLEMENTATION_ORDER
        for seq_len in experiment.DEFAULT_SEQ_LENS
    }


def format_key(key: tuple[str, int]) -> str:
    """Format one configuration for diagnostics."""
    return f"{key[0]}/N{key[1]}"


def read_summary(summary_path: Path) -> list[dict[str, str]]:
    """Read and minimally validate an attention summary CSV."""
    try:
        source = summary_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise PlotError(f"Cannot open summary CSV {summary_path}: {exc}") from exc
    with source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        missing = sorted(SUMMARY_FIELDS - fields)
        if missing:
            raise PlotError(
                "Summary CSV is missing required column(s): "
                + ", ".join(missing)
                + "; rerun the attention cases with --rerun"
            )
        return list(reader)


def parse_metric(
    row: dict[str, str],
    field: str,
    issues: list[str],
    *,
    allow_zero: bool = False,
) -> float:
    """Parse a finite metric and append any diagnostic."""
    try:
        value = float(row[field])
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError
        return value
    except (KeyError, TypeError, ValueError):
        issues.append(f"{field}={row.get(field)!r}")
        return math.nan


def parse_timed_validation_failure(value: str | None) -> bool:
    """Parse the aggregate marker while accepting legacy successful blanks."""
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"", "false", "0"}:
        return False
    raise ValueError(f"invalid boolean {value!r}")


def provenance_issues(
    row: dict[str, str], implementation: str, device: str
) -> list[str]:
    """Return device, mapping, and timing provenance problems."""
    config = experiment.base.device_config(device)
    case = experiment.AttentionCase(
        implementation, experiment.DEFAULT_SEQ_LENS[0], device
    )
    expected = {
        "device": device,
        "target": config.target,
        "backend_target": config.allo_device_type(case.mapping_columns),
        "npu2": config.npu2,
        "dtype": experiment.DTYPE,
        "head_dim": str(experiment.HEAD_DIM),
        "q_chunk_size": str(experiment.Q_CHUNK_SIZE),
        "kv_chunk_size": str(experiment.KV_CHUNK_SIZE),
        "mapping_rows": str(case.mapping_rows),
        "mapping_columns": str(case.mapping_columns),
        "compute_slots": str(case.compute_slots),
        "kernel_count": str(case.kernel_count),
        "timing_scope": experiment.TIMING_SCOPE,
        "attention_timing_version": str(experiment.ATTENTION_TIMING_VERSION),
        "npu_timing_scope": experiment.NPU_TIMING_SCOPE,
        "npu_aggregation": case.npu_aggregation,
    }
    issues = []
    for field, expected_value in expected.items():
        actual = str(row.get(field, "") or "").strip()
        if actual != str(expected_value):
            issues.append(f"{field}={row.get(field)!r}")
    return issues


def validate_summary(
    rows: Sequence[dict[str, str]], device: str = experiment.base.DEFAULT_DEVICE
) -> list[ResultPoint]:
    """Validate a complete selected-device sweep and return parsed points."""
    experiment.base.device_config(device)
    expected = expected_keys()
    seen: dict[tuple[str, int], list[int]] = {}
    points = []
    issues = []
    for row_number, row in enumerate(rows, start=2):
        try:
            key = (row["implementation"], int(row["seq_len"]))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"row {row_number}: malformed configuration ({exc})")
            continue
        seen.setdefault(key, []).append(row_number)
        name = format_key(key)
        if key not in expected:
            issues.append(f"{name}: unexpected configuration")
            continue

        row_issues = provenance_issues(row, key[0], device)
        try:
            timed_failure = parse_timed_validation_failure(
                row.get("timed_validation_failure")
            )
        except ValueError:
            timed_failure = False
            row_issues.append(
                "timed_validation_failure=" f"{row.get('timed_validation_failure')!r}"
            )

        infeasible = key == (
            "baseline",
            experiment.INFEASIBLE_BASELINE_SEQ_LEN,
        )
        metric_fields = (
            "filtered_mean_us",
            "filtered_min_us",
            "filtered_max_us",
            "filtered_npu_mean_us",
            "filtered_extra_mean_us",
        )
        if infeasible:
            if timed_failure:
                row_issues.append("timed_validation_failure must be false")
            if row.get("status") != "infeasible":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "not_run":
                row_issues.append(f"validation={row.get('validation')!r}")
            if any(str(row.get(field, "")).strip() for field in metric_fields):
                row_issues.append("infeasible timing metrics must be empty")
            if row_issues:
                issues.append(f"{name}: " + ", ".join(row_issues))
                continue
            points.append(
                ResultPoint(key[0], key[1], None, None, None, None, None, True)
            )
            continue

        if timed_failure:
            if key[0] != "baseline":
                row_issues.append("timed validation failure requires baseline")
            if row.get("status") != "failed":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "failed":
                row_issues.append(f"validation={row.get('validation')!r}")
        else:
            if row.get("status") != "success":
                row_issues.append(f"status={row.get('status')!r}")
            if row.get("validation") != "passed":
                row_issues.append(f"validation={row.get('validation')!r}")

        e2e_mean = parse_metric(row, "filtered_mean_us", row_issues)
        e2e_min = parse_metric(row, "filtered_min_us", row_issues)
        e2e_max = parse_metric(row, "filtered_max_us", row_issues)
        npu_mean = parse_metric(row, "filtered_npu_mean_us", row_issues)
        extra_mean = parse_metric(
            row, "filtered_extra_mean_us", row_issues, allow_zero=True
        )
        metrics = (e2e_min, e2e_mean, e2e_max, npu_mean, extra_mean)
        if all(math.isfinite(value) for value in metrics):
            if not e2e_min <= e2e_mean <= e2e_max:
                row_issues.append(
                    "filtered E2E metrics must satisfy min <= mean <= max"
                )
            if npu_mean > e2e_mean:
                row_issues.append("filtered NPU mean exceeds E2E mean")
            if not math.isclose(
                npu_mean + extra_mean,
                e2e_mean,
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                row_issues.append("filtered component means do not sum to E2E")
        if row_issues:
            issues.append(f"{name}: " + ", ".join(row_issues))
            continue
        points.append(
            ResultPoint(
                key[0],
                key[1],
                e2e_mean,
                e2e_min,
                e2e_max,
                npu_mean,
                extra_mean,
            )
        )

    for key, row_numbers in seen.items():
        if len(row_numbers) > 1:
            values = ", ".join(str(value) for value in row_numbers)
            issues.append(f"{format_key(key)}: duplicate rows {values}")
    for key in sorted(expected - seen.keys()):
        issues.append(f"{format_key(key)}: missing")
    if issues:
        details = "\n".join(f"  - {issue}" for issue in sorted(issues))
        raise PlotError(
            f"Summary CSV does not contain a complete, valid {len(expected)}-case "
            f"attention sweep for {device}:\n{details}"
        )
    return sorted(
        points,
        key=lambda point: (
            experiment.IMPLEMENTATION_ORDER.index(point.implementation),
            point.seq_len,
        ),
    )


def point_map(points: Sequence[ResultPoint]) -> dict[tuple[str, int], ResultPoint]:
    """Index validated result points by implementation and sequence length."""
    return {(point.implementation, point.seq_len): point for point in points}


def create_figure(points: Sequence[ResultPoint]):
    """Create the grouped stacked E2E timing-breakdown figure."""
    indexed = point_map(points)
    positions = np.arange(len(experiment.DEFAULT_SEQ_LENS), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.0))
    infeasible_positions = []

    for index, implementation in enumerate(experiment.IMPLEMENTATION_ORDER):
        selected = [
            indexed[(implementation, seq_len)]
            for seq_len in experiment.DEFAULT_SEQ_LENS
        ]
        feasible_positions = np.asarray(
            [
                positions[position]
                for position, point in enumerate(selected)
                if not point.infeasible
            ]
        )
        feasible = [point for point in selected if not point.infeasible]
        offset = (index - 0.5) * width
        x_values = feasible_positions + offset
        npu_means = np.asarray([point.npu_mean_us / 1000.0 for point in feasible])
        extra_means = np.asarray([point.extra_mean_us / 1000.0 for point in feasible])
        label = IMPLEMENTATION_LEGEND_LABELS[implementation]
        colors = IMPLEMENTATION_COMPONENT_COLORS[implementation]
        axis.bar(
            x_values,
            npu_means,
            width,
            label=label,
            color=colors["npu"],
        )
        axis.bar(
            x_values,
            extra_means,
            width,
            bottom=npu_means,
            label="_nolegend_",
            color=colors["extra"],
        )
        infeasible_positions.extend(
            positions[position] + offset
            for position, point in enumerate(selected)
            if point.infeasible
        )

    if infeasible_positions:
        x_values = np.asarray(infeasible_positions)
        axis.scatter(
            x_values,
            np.full(x_values.shape, 0.08),
            marker="x",
            color=INFEASIBLE_COLOR,
            s=55,
            linewidths=2,
            transform=axis.get_xaxis_transform(),
            label="_nolegend_",
            zorder=5,
        )

    labels = [str(value) for value in experiment.DEFAULT_SEQ_LENS]
    axis.set_xticks(positions, labels)
    axis.set_xlabel("Sequence Length")
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Attention End-to-End Latency Breakdown")
    axis.set_axisbelow(True)
    axis.grid(True, axis="y", color="lightgray", linestyle="--", alpha=0.7)
    legend_order = ["Unfused", "Fused"]
    handles, handle_labels = axis.get_legend_handles_labels()
    handle_by_label = dict(zip(handle_labels, handles))
    axis.legend(
        [handle_by_label[label] for label in legend_order],
        legend_order,
        loc="upper left",
        ncol=2,
    )
    figure.tight_layout()
    return figure


def render_png(points: Sequence[ResultPoint]) -> bytes:
    """Render the attention stacked bar chart to PNG bytes."""
    figure = create_figure(points)
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


def generate_plot(
    summary_path: Path,
    output_dir: Path,
    device: str = experiment.base.DEFAULT_DEVICE,
) -> Path:
    """Validate a complete sweep and write its stacked bar chart."""
    points = validate_summary(read_summary(summary_path), device)
    image = render_png(points)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "attention_e2e.png"
    atomic_write_bytes(output_path, image)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the plotting command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot stacked end-to-end attention timing breakdowns",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the attention plotting CLI."""
    args = build_parser().parse_args(argv)
    summary = args.summary or default_summary(args.device)
    output_dir = args.output_dir or default_output_dir(args.device)
    try:
        path = generate_plot(summary, output_dir, args.device)
    except PlotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
