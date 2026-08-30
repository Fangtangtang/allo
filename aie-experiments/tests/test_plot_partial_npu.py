# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partial-NPU bf16 GEMM result plotter."""

import csv
import importlib.util
import itertools
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pytest

EXPERIMENT_DIR = Path(__file__).parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
MODULE_PATH = EXPERIMENT_DIR / "plot_partial_npu.py"
SPEC = importlib.util.spec_from_file_location(
    "aie_experiments_plot_partial_npu", MODULE_PATH
)
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)

SUMMARY_FIELDS = [
    "device",
    "target",
    "backend_target",
    "npu2",
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
]


def complete_rows(device="xdna1"):
    """Return a minimal complete physical device sweep summary."""
    rows = []
    sizes = plot.base_plot.DEFAULT_SIZES
    matrix_ns = plot.experiment.default_matrix_ns(device)
    config = plot.experiment.base.device_config(device)
    for columns in plot.experiment.default_columns(device):
        variants = plot.experiment.physical_variants(
            columns, plot.experiment.VARIANT_ORDER, device
        )
        for variant, M, N, K in itertools.product(variants, sizes, matrix_ns, sizes):
            mapping_rows, mapping_columns = plot.expected_mapping_shape(
                variant, columns
            )
            device_columns = plot.expected_device_columns(variant, columns, device)
            flow = "mlir-aie" if variant == "manual" else "allo"
            filtered_gflops = 1000.0 + columns * 100.0 + M + N + K
            mean_us = 2.0 * M * N * K / (filtered_gflops * 1000.0)
            rows.append(
                {
                    "device": device,
                    "target": config.target,
                    "backend_target": config.backend_target(flow, device_columns),
                    "npu2": config.npu2,
                    "variant": variant,
                    "plot_series": ";".join(
                        plot.expected_series(variant, columns, device)
                    ),
                    "flow": flow,
                    "status": "success",
                    "validation": "passed",
                    "timed_validation_failure": False,
                    "dtype": "bf16",
                    "M": M,
                    "N": N,
                    "K": K,
                    "compute_columns": columns,
                    "mapping_columns": mapping_columns,
                    "mapping_rows": mapping_rows,
                    "device_columns": device_columns,
                    "filtered_gflops": filtered_gflops,
                    "filtered_min_us": mean_us * 0.9,
                    "filtered_max_us": mean_us * 1.1,
                }
            )
    return rows


def write_summary(path, rows):
    """Write partial-NPU summary rows."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_performance_envelope_selects_maximum_for_repeated_intensity():
    points = [
        plot.ResultPoint("manual", 1, 256, 256, 256, 0.5, 0.4, 0.6),
        plot.ResultPoint("manual", 1, 256, 256, 512, 1.0, 1.8, 3.0),
        plot.ResultPoint("manual", 1, 512, 256, 256, 2.0, 1.5, 2.5),
    ]
    intensities, tops, lower_tops, upper_tops = plot.performance_envelope(
        points, "manual"
    )
    assert intensities == sorted(intensities)
    assert tops == [0.5, 2.0]
    assert lower_tops == [0.4, 1.8]
    assert upper_tops == [0.6, 3.0]


def test_four_column_canonical_allo_rows_supply_both_logical_series():
    points = plot.validate_summary(complete_rows(), 4)
    assert len(points) == 192
    by_series = {
        series: [point for point in points if point.series == series]
        for series in plot.SERIES_ORDER
    }
    assert all(len(series_points) == 64 for series_points in by_series.values())

    compiled = {
        (point.M, point.N, point.K): (point.tops, point.lower_tops, point.upper_tops)
        for point in by_series["compiled"]
    }
    full_io = {
        (point.M, point.N, point.K): (point.tops, point.lower_tops, point.upper_tops)
        for point in by_series["compiled-full-io"]
    }
    assert compiled == full_io


def test_complete_summary_generates_three_styled_plots(tmp_path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, complete_rows())

    paths, skipped = plot.generate_plots(summary_path, output_dir)
    assert not skipped
    assert [path.name for path in paths] == [
        "gemm_bf16_1x4.png",
        "gemm_bf16_2x4.png",
        "gemm_bf16_4x4.png",
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)

    points = plot.validate_summary(plot.read_summary(summary_path), 4)
    figure = plot.create_figure(points, 4)
    try:
        axis = figure.axes[0]
        assert [line.get_label() for line in axis.lines] == [
            "Manual Template",
            "Compiled",
            "Compiled (Full I/O)",
        ]
        assert [line.get_color() for line in axis.lines] == [
            plot.SERIES_COLORS[series] for series in plot.SERIES_ORDER
        ]
        assert axis.lines[2].get_linestyle() == "--"
        assert len(axis.collections) == 3
        assert axis.get_xlabel() == "Arithmetic Intensity (OPs/byte)"
        assert axis.get_ylabel() == "Performance (TOP/s)"
        assert axis.get_xlim()[0] == 0
        assert axis.get_ylim()[0] == 0
        legend_state = vars(axis.get_legend())
        assert legend_state.get("_loc_real", legend_state.get("_loc")) == 4
    finally:
        plt.close(figure)


def test_incomplete_column_is_skipped_without_blocking_other_plots(tmp_path):
    rows = complete_rows()
    rows.remove(next(row for row in rows if row["compute_columns"] == 2))
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)

    paths, skipped = plot.generate_plots(summary_path, output_dir)
    assert [path.name for path in paths] == [
        "gemm_bf16_1x4.png",
        "gemm_bf16_4x4.png",
    ]
    assert list(skipped) == [2]
    assert "missing" in skipped[2]
    assert not (output_dir / "gemm_bf16_2x4.png").exists()


def test_marked_allo_failures_are_plotted_normally(tmp_path):
    rows = complete_rows()
    for row in rows:
        if row["flow"] == "allo":
            row["status"] = "failed"
            row["validation"] = "failed"
            row["timed_validation_failure"] = True

    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)
    paths, skipped = plot.generate_plots(summary_path, output_dir)

    assert len(paths) == 3
    assert not skipped


def test_no_complete_column_writes_no_images(tmp_path):
    rows = complete_rows()
    for row in rows:
        row["status"] = "failed"

    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots"
    write_summary(summary_path, rows)
    with pytest.raises(plot.PlotError, match="No compute-width"):
        plot.generate_plots(summary_path, output_dir)
    assert not output_dir.exists()


def test_xdna2_summary_generates_four_plots_and_validates_provenance(tmp_path):
    rows = complete_rows("xdna2")
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots" / "xdna2"
    write_summary(summary_path, rows)

    paths, skipped = plot.generate_plots(summary_path, output_dir, device="xdna2")
    assert not skipped
    assert [path.name for path in paths] == [
        "gemm_bf16_1x4.png",
        "gemm_bf16_2x4.png",
        "gemm_bf16_4x4.png",
        "gemm_bf16_8x4.png",
    ]
    points = plot.validate_summary(plot.read_summary(summary_path), 8, "xdna2")
    assert len(points) == 144
    assert all(
        sum(point.series == series for point in points) == 48
        for series in plot.SERIES_ORDER
    )
    assert plot.default_summary("xdna2").parent.name == "gemm-partial-npu-xdna2"
    assert plot.default_output_dir("xdna2").name == "xdna2"

    target = next(row for row in rows if row["compute_columns"] == 8)
    target["backend_target"] = "npu"
    write_summary(summary_path, rows)
    mismatch_dir = tmp_path / "mismatch"
    paths, skipped = plot.generate_plots(summary_path, mismatch_dir, device="xdna2")
    assert [path.name for path in paths] == [
        "gemm_bf16_1x4.png",
        "gemm_bf16_2x4.png",
        "gemm_bf16_4x4.png",
    ]
    assert "backend_target='npu'" in skipped[8]
