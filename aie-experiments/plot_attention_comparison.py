#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot XDNA1 and XDNA2 attention timing breakdowns side by side."""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import plot_attention as attention_plot  # pylint: disable=wrong-import-position

DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "plots"
OUTPUT_NAME = "attention_e2e_comparison.png"
DEVICE_TITLES = {"xdna1": "XDNA1", "xdna2": "XDNA2"}
LEGEND_ORDER = ["Unfused", "Fused"]


def load_points(summary_path: Path, device: str):
    """Load and validate one device's attention summary."""
    try:
        rows = attention_plot.read_summary(summary_path)
        return attention_plot.validate_summary(rows, device)
    except attention_plot.PlotError as exc:
        raise attention_plot.PlotError(
            f"{DEVICE_TITLES[device]} summary {summary_path}: {exc}"
        ) from exc


def create_figure(xdna1_points, xdna2_points):
    """Create the two-panel XDNA attention comparison figure."""
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 3), sharey=False)
    for axis, device, points in zip(
        axes,
        ("xdna1", "xdna2"),
        (xdna1_points, xdna2_points),
    ):
        attention_plot.draw_attention_axis(
            axis,
            points,
            title=DEVICE_TITLES[device],
            show_legend=False,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    figure.suptitle("Attention Kernel End-to-End Latency Breakdown")
    figure.legend(
        [handle_by_label[label] for label in LEGEND_ORDER],
        LEGEND_ORDER,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),
        ncol=2,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def render_png(xdna1_points, xdna2_points) -> bytes:
    """Render the two-panel attention chart to PNG bytes."""
    figure = create_figure(xdna1_points, xdna2_points)
    output = io.BytesIO()
    try:
        figure.savefig(output, format="png", dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)
    return output.getvalue()


def generate_plot(
    xdna1_summary: Path,
    xdna2_summary: Path,
    output_dir: Path,
) -> Path:
    """Validate both device sweeps and write their comparison chart."""
    xdna1_points = load_points(xdna1_summary, "xdna1")
    xdna2_points = load_points(xdna2_summary, "xdna2")
    image = render_png(xdna1_points, xdna2_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    attention_plot.atomic_write_bytes(output_path, image)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison plotting command-line parser."""
    parser = argparse.ArgumentParser(
        description="Plot XDNA1 and XDNA2 attention timing breakdowns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xdna1-summary",
        type=Path,
        default=attention_plot.default_summary("xdna1"),
        help="XDNA1 summary CSV",
    )
    parser.add_argument(
        "--xdna2-summary",
        type=Path,
        default=attention_plot.default_summary("xdna2"),
        help="XDNA2 summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="plot directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the two-device attention plotting CLI."""
    args = build_parser().parse_args(argv)
    try:
        path = generate_plot(
            args.xdna1_summary,
            args.xdna2_summary,
            args.output_dir,
        )
    except attention_plot.PlotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
