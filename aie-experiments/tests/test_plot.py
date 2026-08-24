# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NPU GEMM result plotter."""

import csv
import importlib.util
import itertools
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest

MODULE_PATH = Path(__file__).parents[1] / "plot.py"
SPEC = importlib.util.spec_from_file_location("aie_experiments_plot", MODULE_PATH)
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)

SUMMARY_FIELDS = [
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
]


def complete_rows():
    """Return a minimal, valid default-sweep summary."""
    rows = []
    for flow, dtype, M, N, K in itertools.product(
        plot.FLOW_ORDER,
        plot.DTYPE_ORDER,
        plot.DEFAULT_SIZES,
        plot.DEFAULT_SIZES,
        plot.DEFAULT_SIZES,
    ):
        filtered_gflops = 1000.0 + M + N + K
        mean_us = 2.0 * M * N * K / (filtered_gflops * 1000.0)
        rows.append(
            {
                "flow": flow,
                "status": "success",
                "validation": "passed",
                "timed_validation_failure": False,
                "dtype": dtype,
                "M": M,
                "N": N,
                "K": K,
                "filtered_gflops": filtered_gflops,
                "filtered_min_us": mean_us * 0.9,
                "filtered_max_us": mean_us * 1.1,
            }
        )
    return rows


def write_summary(path, rows):
    """Write test summary rows."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_arithmetic_intensity_uses_dtype_bytes():
    assert plot.arithmetic_intensity(256, 256, 256, "int16") == pytest.approx(
        85.3333333333
    )
    assert plot.arithmetic_intensity(256, 256, 256, "bf16") == pytest.approx(
        85.3333333333
    )
    assert plot.arithmetic_intensity(256, 256, 256, "int8") == pytest.approx(
        170.6666666667
    )


def test_performance_envelope_selects_maximum_for_repeated_intensity():
    points = [
        plot.ResultPoint("allo", "int16", 256, 256, 256, 0.5, 0.4, 0.6),
        plot.ResultPoint("allo", "int16", 256, 256, 512, 1.0, 1.8, 3.0),
        plot.ResultPoint("allo", "int16", 512, 256, 256, 2.0, 1.5, 2.5),
    ]
    intensities, tops, lower_tops, upper_tops = plot.performance_envelope(
        points, "allo", "int16"
    )
    assert intensities == sorted(intensities)
    assert tops == [0.5, 2.0]
    assert lower_tops == [0.4, 1.8]
    assert upper_tops == [0.6, 3.0]


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "duplicate rows"),
        ("failed", "status='failed'"),
        ("unvalidated", "validation='failed'"),
        ("invalid_metric", "filtered_gflops=''"),
        ("invalid_min", "filtered_min_us=''"),
        ("invalid_max", "filtered_max_us=''"),
    ],
)
def test_invalid_datatype_is_skipped_while_complete_datatypes_are_plotted(
    tmp_path, problem, message
):
    rows = complete_rows()
    target = next(row for row in rows if row["dtype"] == "bf16")
    if problem == "missing":
        rows.remove(target)
    elif problem == "duplicate":
        rows.append(target.copy())
    elif problem == "failed":
        target["status"] = "failed"
    elif problem == "unvalidated":
        target["validation"] = "failed"
    elif problem == "invalid_metric":
        target["filtered_gflops"] = ""
    elif problem == "invalid_min":
        target["filtered_min_us"] = ""
    else:
        target["filtered_max_us"] = ""

    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)
    paths, skipped = plot.generate_plots(summary_path, output_dir)
    assert [path.name for path in paths] == ["gemm_int16.png", "gemm_int8.png"]
    assert list(skipped) == ["bf16"]
    assert message in skipped["bf16"]


def test_no_complete_datatype_writes_no_plots(tmp_path):
    rows = complete_rows()
    for row in rows:
        row["status"] = "failed"

    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)
    with pytest.raises(plot.PlotError, match="No datatype"):
        plot.generate_plots(summary_path, output_dir)
    assert not output_dir.exists()


def test_complete_summary_generates_three_plots_with_two_curves(tmp_path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, complete_rows())

    paths, skipped = plot.generate_plots(summary_path, output_dir)
    assert not skipped
    assert [path.name for path in paths] == [
        "gemm_int16.png",
        "gemm_int8.png",
        "gemm_bf16.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)

    points = plot.validate_summary(plot.read_summary(summary_path))
    for dtype in plot.DTYPE_ORDER:
        figure = plot.create_figure(points, dtype)
        try:
            assert [line.get_label() for line in figure.axes[0].lines] == [
                plot.FLOW_LABELS[flow] for flow in plot.FLOW_ORDER
            ]
            assert len(figure.axes[0].collections) == 2
            assert figure.axes[0].get_xlabel() == "Arithmetic Intensity (OPs/byte)"
            assert figure.axes[0].get_ylabel() == "Performance (TOP/s)"
            legend_state = vars(figure.axes[0].get_legend())
            assert legend_state.get("_loc_real", legend_state.get("_loc")) == 4
        finally:
            plt.close(figure)


def test_marked_allo_bf16_failures_are_plotted_normally(tmp_path):
    rows = complete_rows()
    for row in rows:
        if row["flow"] == "allo" and row["dtype"] == "bf16":
            row["status"] = "failed"
            row["validation"] = "failed"
            row["timed_validation_failure"] = True

    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)
    paths, skipped = plot.generate_plots(summary_path, output_dir)

    assert not skipped
    assert [path.name for path in paths] == [
        "gemm_int16.png",
        "gemm_int8.png",
        "gemm_bf16.png",
    ]
    points = plot.validate_summary(plot.read_summary(summary_path), ["bf16"])
    figure = plot.create_figure(points, "bf16")
    try:
        assert [line.get_label() for line in figure.axes[0].lines] == [
            plot.FLOW_LABELS[flow] for flow in plot.FLOW_ORDER
        ]
        assert len(figure.axes[0].collections) == 2
    finally:
        plt.close(figure)
