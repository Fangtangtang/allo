# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the two-device attention comparison plotter."""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

MODULE_PATH = Path(__file__).parents[1] / "plot_attention_comparison.py"
SPEC = importlib.util.spec_from_file_location(
    "aie_experiments_plot_attention_comparison", MODULE_PATH
)
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


def device_points(device, scale):
    """Return synthetic validated result points for one device."""
    points = []
    for implementation in plot.attention_plot.experiment.IMPLEMENTATION_ORDER:
        implementation_scale = 1.0 if implementation == "baseline" else 0.5
        for seq_len in plot.attention_plot.experiment.DEFAULT_SEQ_LENS:
            case = plot.attention_plot.experiment.AttentionCase(
                implementation, seq_len, device
            )
            if case.infeasible:
                points.append(
                    plot.attention_plot.ResultPoint(
                        implementation,
                        seq_len,
                        None,
                        None,
                        None,
                        None,
                        None,
                        True,
                    )
                )
                continue
            e2e_mean = scale * implementation_scale * seq_len
            npu_mean = e2e_mean * 0.25
            points.append(
                plot.attention_plot.ResultPoint(
                    implementation,
                    seq_len,
                    e2e_mean,
                    e2e_mean * 0.8,
                    e2e_mean * 1.2,
                    npu_mean,
                    e2e_mean - npu_mean,
                )
            )
    return points


def test_create_figure_has_independent_device_panels():
    xdna1_points = device_points("xdna1", 10.0)
    xdna2_points = device_points("xdna2", 2.0)

    figure = plot.create_figure(xdna1_points, xdna2_points)
    try:
        assert len(figure.axes) == 2
        xdna1_axis, xdna2_axis = figure.axes
        assert [axis.get_title() for axis in figure.axes] == ["XDNA1", "XDNA2"]
        assert all(axis.get_xlabel() == "Sequence Length" for axis in figure.axes)
        assert all(axis.get_ylabel() == "Latency (ms)" for axis in figure.axes)
        assert all(axis.get_yscale() == "linear" for axis in figure.axes)
        assert not xdna1_axis.get_shared_y_axes().joined(xdna1_axis, xdna2_axis)
        assert xdna1_axis.get_ylim()[1] > xdna2_axis.get_ylim()[1]

        assert len(xdna1_axis.patches) == 22
        assert len(xdna2_axis.patches) == 20
        baseline_count = sum(
            point.implementation == "baseline" and not point.infeasible
            for point in xdna1_points
        )
        assert xdna1_axis.patches[baseline_count].get_y() == pytest.approx(
            xdna1_axis.patches[0].get_height()
        )

        xdna1_cross = next(
            item for item in xdna1_axis.collections if item.get_label() == "_nolegend_"
        )
        xdna2_cross = next(
            item for item in xdna2_axis.collections if item.get_label() == "_nolegend_"
        )
        assert xdna1_cross.get_offsets().shape == (1, 2)
        assert xdna2_cross.get_offsets().shape == (2, 2)

        assert len(figure.legends) == 1
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Unfused",
            "Fused",
        ]
        assert all(axis.get_legend() is None for axis in figure.axes)
    finally:
        plt.close(figure)


def test_defaults_and_cli_overrides(monkeypatch, tmp_path, capsys):
    args = plot.build_parser().parse_args([])
    assert args.xdna1_summary == plot.attention_plot.default_summary("xdna1")
    assert args.xdna2_summary == plot.attention_plot.default_summary("xdna2")
    assert args.output_dir == plot.DEFAULT_OUTPUT_DIR

    summaries = [tmp_path / "one.csv", tmp_path / "two.csv"]
    output_dir = tmp_path / "figures"
    calls = []

    def fake_generate(xdna1_summary, xdna2_summary, selected_output_dir):
        calls.append((xdna1_summary, xdna2_summary, selected_output_dir))
        return selected_output_dir / plot.OUTPUT_NAME

    monkeypatch.setattr(plot, "generate_plot", fake_generate)
    result = plot.main(
        [
            "--xdna1-summary",
            str(summaries[0]),
            "--xdna2-summary",
            str(summaries[1]),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 0
    assert calls == [(summaries[0], summaries[1], output_dir)]
    assert capsys.readouterr().out.strip() == str(output_dir / plot.OUTPUT_NAME)


def test_load_points_identifies_invalid_device_summary(monkeypatch, tmp_path):
    summary_path = tmp_path / "bad.csv"

    def fail_read(_path):
        raise plot.attention_plot.PlotError("broken input")

    monkeypatch.setattr(plot.attention_plot, "read_summary", fail_read)

    with pytest.raises(
        plot.attention_plot.PlotError,
        match=r"XDNA2 summary .*bad\.csv: broken input",
    ):
        plot.load_points(summary_path, "xdna2")


def test_generate_plot_creates_atomic_png(monkeypatch, tmp_path):
    xdna1_summary = tmp_path / "xdna1.csv"
    xdna2_summary = tmp_path / "xdna2.csv"
    output_dir = tmp_path / "plots"
    calls = []

    def fake_load(summary_path, device):
        calls.append((summary_path, device))
        scale = 10.0 if device == "xdna1" else 2.0
        return device_points(device, scale)

    monkeypatch.setattr(plot, "load_points", fake_load)

    output_path = plot.generate_plot(xdna1_summary, xdna2_summary, output_dir)

    assert calls == [
        (xdna1_summary, "xdna1"),
        (xdna2_summary, "xdna2"),
    ]
    assert output_path == output_dir / "attention_e2e_comparison.png"
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert not list(output_dir.glob(".*.tmp-*"))
