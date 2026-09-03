# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the attention end-to-end stacked bar plotter."""

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.container import BarContainer, ErrorbarContainer

MODULE_PATH = Path(__file__).parents[1] / "plot_attention.py"
SPEC = importlib.util.spec_from_file_location(
    "aie_experiments_plot_attention", MODULE_PATH
)
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)

SUMMARY_FIELDS = sorted(plot.SUMMARY_FIELDS)


def complete_rows(device="xdna1"):
    """Return a complete synthetic attention summary."""
    config = plot.experiment.base.device_config(device)
    rows = []
    for implementation in plot.experiment.IMPLEMENTATION_ORDER:
        for seq_len in plot.experiment.DEFAULT_SEQ_LENS:
            case = plot.experiment.AttentionCase(implementation, seq_len, device)
            e2e_mean = seq_len * (10.0 if implementation == "baseline" else 5.0)
            npu_mean = e2e_mean * 0.25
            infeasible = case.infeasible
            rows.append(
                {
                    "device": device,
                    "target": config.target,
                    "backend_target": config.allo_device_type(case.mapping_columns),
                    "npu2": config.npu2,
                    "implementation": implementation,
                    "status": "infeasible" if infeasible else "success",
                    "validation": "not_run" if infeasible else "passed",
                    "timed_validation_failure": False,
                    "dtype": plot.experiment.DTYPE,
                    "seq_len": seq_len,
                    "head_dim": plot.experiment.HEAD_DIM,
                    "q_chunk_size": plot.experiment.Q_CHUNK_SIZE,
                    "kv_chunk_size": plot.experiment.KV_CHUNK_SIZE,
                    "mapping_rows": case.mapping_rows,
                    "mapping_columns": case.mapping_columns,
                    "compute_slots": case.compute_slots,
                    "kernel_count": case.kernel_count,
                    "timing_scope": plot.experiment.TIMING_SCOPE,
                    "attention_timing_version": (
                        plot.experiment.ATTENTION_TIMING_VERSION
                    ),
                    "npu_timing_scope": plot.experiment.NPU_TIMING_SCOPE,
                    "npu_aggregation": case.npu_aggregation,
                    "filtered_mean_us": "" if infeasible else e2e_mean,
                    "filtered_min_us": ("" if infeasible else e2e_mean * 0.8),
                    "filtered_max_us": ("" if infeasible else e2e_mean * 1.2),
                    "filtered_npu_mean_us": ("" if infeasible else npu_mean),
                    "filtered_extra_mean_us": (
                        "" if infeasible else e2e_mean - npu_mean
                    ),
                }
            )
    return rows


def write_summary(path, rows, fieldnames=SUMMARY_FIELDS):
    """Write synthetic summary rows."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fieldnames} for row in rows
        )


def test_complete_summary_creates_grouped_stacked_layout(tmp_path):
    summary_path = tmp_path / "summary.csv"
    rows = complete_rows("xdna2")
    rows[0].update(
        status="failed",
        validation="failed",
        timed_validation_failure=True,
    )
    write_summary(summary_path, rows)
    points = plot.validate_summary(plot.read_summary(summary_path), "xdna2")

    assert len(points) == 12
    first = points[0]
    assert first.e2e_mean_us == 640.0
    assert first.npu_mean_us == 160.0
    assert first.extra_mean_us == 480.0
    assert points[4].infeasible
    assert points[5].infeasible

    figure = plot.create_figure(points)
    try:
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_yscale() == "linear"
        assert axis.get_ylabel() == "Latency (ms)"
        assert axis.get_title() == "Attention End-to-End Latency Breakdown"
        assert len(axis.patches) == 20

        baseline_npu = axis.patches[:4]
        baseline_extra = axis.patches[4:8]
        flash_npu = axis.patches[8:14]
        flash_extra = axis.patches[14:20]
        assert baseline_npu[0].get_height() == pytest.approx(0.16)
        assert baseline_extra[0].get_y() == pytest.approx(0.16)
        assert baseline_npu[0].get_height() + baseline_extra[
            0
        ].get_height() == pytest.approx(0.64)
        assert flash_npu[0].get_height() + flash_extra[0].get_height() == pytest.approx(
            0.32
        )

        assert baseline_npu[0].get_facecolor() != baseline_extra[0].get_facecolor()
        assert flash_npu[0].get_facecolor() != flash_extra[0].get_facecolor()
        assert baseline_npu[0].get_hatch() is None
        assert flash_npu[0].get_hatch() is None
        assert baseline_extra[0].get_hatch() is None
        assert flash_extra[0].get_hatch() is None

        bar_containers = [
            item for item in axis.containers if isinstance(item, BarContainer)
        ]
        error_containers = [
            item for item in axis.containers if isinstance(item, ErrorbarContainer)
        ]
        assert len(bar_containers) == 4
        assert len(error_containers) == 0

        cross = next(
            item for item in axis.collections if item.get_label() == "_nolegend_"
        )
        assert cross.get_offsets().shape == (2, 2)
        expected_color = plot.matplotlib.colors.to_rgba(plot.INFEASIBLE_COLOR)
        assert cross.get_edgecolors()[0] == pytest.approx(expected_color)
        assert [text.get_text() for text in axis.get_xticklabels()] == [
            str(value) for value in plot.experiment.DEFAULT_SEQ_LENS
        ]
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert legend_labels == ["Unfused", "Fused"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "duplicate rows"),
        ("failed", "status='failed'"),
        ("unvalidated", "validation='failed'"),
        ("metric", "filtered_npu_mean_us=''"),
        ("provenance", "mapping_columns='4'"),
        ("components", "component means do not sum"),
        ("npu_too_large", "NPU mean exceeds"),
    ],
)
def test_invalid_sweep_is_rejected(tmp_path, problem, message):
    rows = complete_rows("xdna2")
    target = rows[-1]
    if problem == "missing":
        rows.remove(target)
    elif problem == "duplicate":
        rows.append(target.copy())
    elif problem == "failed":
        target["status"] = "failed"
    elif problem == "unvalidated":
        target["validation"] = "failed"
    elif problem == "metric":
        target["filtered_npu_mean_us"] = ""
    elif problem == "provenance":
        target["mapping_columns"] = "4"
    elif problem == "components":
        target["filtered_extra_mean_us"] += 1.0
    else:
        target["filtered_npu_mean_us"] = target["filtered_mean_us"] + 1.0
    summary_path = tmp_path / "summary.csv"
    write_summary(summary_path, rows)

    with pytest.raises(plot.PlotError, match=message):
        plot.validate_summary(plot.read_summary(summary_path), "xdna2")


def test_legacy_summary_requires_rerun(tmp_path):
    summary_path = tmp_path / "summary.csv"
    fields = sorted(
        plot.SUMMARY_FIELDS - {"attention_timing_version", "filtered_npu_mean_us"}
    )
    write_summary(summary_path, complete_rows(), fields)

    with pytest.raises(plot.PlotError, match="rerun"):
        plot.read_summary(summary_path)


def test_complete_summary_generates_atomic_png(tmp_path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "plots" / "xdna2"
    write_summary(summary_path, complete_rows("xdna2"))

    output_path = plot.generate_plot(summary_path, output_dir, "xdna2")

    assert output_path == output_dir / "attention_e2e.png"
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(output_dir.glob(".*.tmp-*"))


def test_device_specific_defaults():
    assert plot.default_summary("xdna1").parent.name == "attention"
    assert plot.default_summary("xdna2").parent.name == "attention-xdna2"
    assert plot.default_output_dir("xdna1").name == "plots"
    assert plot.default_output_dir("xdna2").name == "xdna2"
