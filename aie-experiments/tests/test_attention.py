# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent tests for the attention experiment runner."""

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "attention.py"
SPEC = importlib.util.spec_from_file_location("aie_experiments_attention", MODULE_PATH)
attention = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = attention
SPEC.loader.exec_module(attention)


def test_default_case_expansion_and_device_mappings():
    xdna1 = attention.generate_cases("both", attention.DEFAULT_SEQ_LENS, "xdna1")
    xdna2 = attention.generate_cases("both", attention.DEFAULT_SEQ_LENS, "xdna2")

    assert len(xdna1) == len(xdna2) == 12
    assert [(case.mapping_rows, case.mapping_columns) for case in xdna1] == [
        (4, 4)
    ] * 12
    assert [(case.mapping_rows, case.mapping_columns) for case in xdna2] == [
        (4, 8)
    ] * 12
    assert [case.kernel_count for case in xdna2[:6]] == [4] * 6
    assert [case.kernel_count for case in xdna2[6:]] == [1] * 6
    assert [case.npu_aggregation for case in xdna2[:6]] == ["sum-of-4-kernels"] * 6
    assert [case.npu_aggregation for case in xdna2[6:]] == ["single-kernel"] * 6
    assert [case.seq_len for case in xdna1 if case.infeasible] == [2048]
    assert [case.seq_len for case in xdna2 if case.infeasible] == [2048]


def test_mapping_helpers_use_full_device_capacity():
    score = attention.gemm_mapping_primitives(1, 16, 16, 4, 8)
    bundles = [nodes for primitive, nodes in score if primitive == "bundle"]
    assert len(bundles) == 32
    assert all(len(nodes) == 8 for nodes in bundles)

    output = attention.gemm_mapping_primitives(32, 32, 1, 32, 1)
    assert sum(primitive == "chain" for primitive, _ in output) == 31 * 32
    assert not any(primitive == "bundle" for primitive, _ in output)

    softmax = attention.linear_mapping_primitives("core", 512, 32)
    assert len(softmax) == 32
    assert all(len(nodes) == 16 for _, nodes in softmax)


def test_softmax_wrapper_is_architecture_specific_and_fixed_shape(tmp_path):
    aie2_top, aie2_path = attention.write_softmax_wrapper(
        64, "xdna1", tmp_path / "aie2.cc"
    )
    aie2p_top, aie2p_path = attention.write_softmax_wrapper(
        2048, "xdna2", tmp_path / "aie2p.cc"
    )

    assert aie2_top == "attention_softmax_bf16_64"
    assert "softmax_bf16.cc" in aie2_path.read_text(encoding="utf-8")
    assert "input[4][64]" in aie2_path.read_text(encoding="utf-8")
    assert aie2p_top == "attention_softmax_bf16_2048"
    assert "softmax_bf16_aie2p.cc" in aie2p_path.read_text(encoding="utf-8")
    assert "softmax_simple_bf16<2048>" in aie2p_path.read_text(encoding="utf-8")


def test_baseline_regions_resolve_module_global_annotations(tmp_path, monkeypatch):
    import allo.dataflow as df

    checked = []

    def customize_only(region, **_kwargs):
        df.customize(region)
        checked.append(region.__name__)
        return object()

    monkeypatch.setattr(df, "build", customize_only)
    modules = attention._build_baseline_modules(
        attention.AttentionCase("baseline", 64, "xdna1"), tmp_path
    )

    assert len(modules) == 4
    assert checked == [
        "score_region",
        "scale_region",
        "softmax_region",
        "output_region",
    ]


def test_flash_selector_restores_environment(monkeypatch):
    monkeypatch.setenv("NPU2", "2")
    with attention.flash_architecture_selector("xdna2"):
        assert os.environ["NPU2"] == "1"
    assert os.environ["NPU2"] == "2"


def test_complete_attention_timing_and_baseline_order():
    events = []
    modules = [lambda value=value: events.append(value) for value in range(4)]
    clock = iter((0, 1000, 2000, 4000))

    samples = attention.measure_complete_attention(
        lambda: attention.run_baseline_once(modules),
        warmup=1,
        iterations=2,
        clock_ns=lambda: next(clock),
    )

    assert samples == [1.0, 2.0]
    assert events == [0, 1, 2, 3] * 3


def test_timing_parser_validates_count_and_values():
    output = (
        f"noise\n{attention.TIMING_PREFIX}10.5\n" f"{attention.TIMING_PREFIX}11.25\n"
    )
    assert attention.parse_sample_timings(output, 2) == [10.5, 11.25]
    with pytest.raises(attention.ExperimentError, match="Expected 1"):
        attention.parse_sample_timings(output, 1)
    with pytest.raises(attention.ExperimentError, match="Invalid"):
        attention.parse_sample_timings(f"{attention.TIMING_PREFIX}0", 1)


def npu_output(values):
    """Return generated-host timing output for the supplied microseconds."""
    return "\n".join(f"{attention.NPU_TIMING_PREFIX}{value}us" for value in values)


def test_npu_timing_parser_aggregates_and_discards_setup_runs():
    baseline = attention.AttentionCase("baseline", 64)
    flash = attention.AttentionCase("flash", 64)

    assert attention.parse_npu_timings(
        npu_output(range(1, 17)), baseline, warmup=1, iterations=2
    ) == [42.0, 58.0]
    assert attention.parse_npu_timings(
        npu_output([10, 20, 30, 40]), flash, warmup=1, iterations=2
    ) == [30.0, 40.0]


def test_npu_timing_parser_rejects_malformed_or_incorrect_output():
    flash = attention.AttentionCase("flash", 64)
    with pytest.raises(attention.ExperimentError, match="Malformed"):
        attention.parse_npu_timings(
            f"{attention.NPU_TIMING_PREFIX}12", flash, warmup=0, iterations=0
        )
    with pytest.raises(attention.ExperimentError, match="Malformed"):
        attention.parse_npu_timings(npu_output(["bad"]), flash, warmup=0, iterations=0)
    with pytest.raises(attention.ExperimentError, match="Invalid"):
        attention.parse_npu_timings(npu_output([0]), flash, warmup=0, iterations=0)
    with pytest.raises(attention.ExperimentError, match="Expected 2"):
        attention.parse_npu_timings(npu_output([1]), flash, warmup=0, iterations=1)
    with pytest.raises(attention.ExperimentError, match="Expected 2"):
        attention.parse_npu_timings(
            npu_output([1, 2, 3]), flash, warmup=0, iterations=1
        )


def test_extra_timing_validation():
    assert attention.derive_extra_timings([10, 12], [3, 5]) == [7.0, 7.0]
    with pytest.raises(attention.ExperimentError, match="count mismatch"):
        attention.derive_extra_timings([10], [])
    with pytest.raises(attention.ExperimentError, match="exceeds E2E"):
        attention.derive_extra_timings([10], [11])


def test_record_processing_and_resume(tmp_path):
    case = attention.AttentionCase("baseline", 64, "xdna2")
    record = attention.new_record(case, 1, 4, tmp_path)
    record.update(
        {
            "status": "success",
            "validation": "passed",
            "timings_us": [10.0, 11.0, 12.0, 100.0],
            "npu_timings_us": [2.0, 3.0, 4.0, 5.0],
            "elapsed_seconds": 1.5,
        }
    )
    path = attention.record_path(tmp_path, case)
    attention.base.atomic_write_json(path, record)

    assert attention.is_resumable(path, attention.case_signature(case, 1, 4))
    raw_path, filtered_path, summary_path = attention.process_results(tmp_path)
    assert raw_path.is_file() and filtered_path.is_file() and summary_path.is_file()
    assert len(raw_path.read_text(encoding="utf-8").splitlines()) == 5
    assert len(filtered_path.read_text(encoding="utf-8").splitlines()) == 4
    with raw_path.open(encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    assert [float(row["npu_time_us"]) for row in raw_rows] == [2, 3, 4, 5]
    assert [float(row["extra_time_us"]) for row in raw_rows] == [8, 8, 8, 95]
    with filtered_path.open(encoding="utf-8") as source:
        filtered_rows = list(csv.DictReader(source))
    assert [float(row["time_us"]) for row in filtered_rows] == [10, 11, 12]
    assert [float(row["npu_time_us"]) for row in filtered_rows] == [2, 3, 4]
    assert [float(row["extra_time_us"]) for row in filtered_rows] == [8, 8, 8]
    with summary_path.open(encoding="utf-8") as source:
        summary = next(csv.DictReader(source))
    assert float(summary["filtered_mean_us"]) == 11.0
    assert float(summary["filtered_npu_mean_us"]) == 3.0
    assert float(summary["filtered_extra_mean_us"]) == 8.0
    assert float(summary["filtered_mean_us"]) == pytest.approx(
        float(summary["filtered_npu_mean_us"])
        + float(summary["filtered_extra_mean_us"])
    )


def test_legacy_or_incomplete_paired_record_is_not_resumable(tmp_path):
    case = attention.AttentionCase("flash", 64)
    record = attention.new_record(case, 0, 1, tmp_path)
    record.update({"status": "success", "validation": "passed", "timings_us": [10.0]})
    record.pop("npu_timings_us")
    record["signature"].pop("attention_timing_version")
    path = attention.record_path(tmp_path, case)
    attention.base.atomic_write_json(path, record)
    assert not attention.is_resumable(path, attention.case_signature(case, 0, 1))
    raw_path, _, summary_path = attention.process_results(tmp_path)
    with raw_path.open(encoding="utf-8") as source:
        legacy_raw = next(csv.DictReader(source))
    assert legacy_raw["npu_time_us"] == ""
    assert legacy_raw["extra_time_us"] == ""
    with summary_path.open(encoding="utf-8") as source:
        legacy_summary = next(csv.DictReader(source))
    assert legacy_summary["filtered_npu_mean_us"] == ""
    assert legacy_summary["filtered_extra_mean_us"] == ""

    record = attention.new_record(case, 0, 1, tmp_path)
    record.update({"status": "success", "validation": "passed", "timings_us": [10.0]})
    attention.base.atomic_write_json(path, record)
    assert not attention.is_resumable(path, attention.case_signature(case, 0, 1))


def test_list_and_dry_run_need_no_hardware(capsys):
    assert attention.main(["list", "--device", "xdna2", "--seq-len", "64"]) == 0
    listed = capsys.readouterr().out
    assert "mapping=4x8" in listed
    assert "Total: 2 configuration(s)" in listed

    assert (
        attention.main(
            [
                "run",
                "--device",
                "xdna2",
                "--implementation",
                "flash",
                "--seq-len",
                "64",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--dry-run",
            ]
        )
        == 0
    )
    preview = capsys.readouterr().out
    assert "--device xdna2" in preview
    assert "--implementation flash" in preview
    assert "Dry run: 1 configuration(s)" in preview


def test_safe_remove_rejects_paths_outside_work_root(tmp_path):
    with pytest.raises(attention.ExperimentError, match="unsafe work path"):
        attention.safe_remove_work(tmp_path)


def test_saved_record_is_json_serializable(tmp_path):
    case = attention.AttentionCase("flash", 128)
    record = attention.new_record(case, 20, 100, tmp_path)
    encoded = json.dumps(record)
    assert '"timing_scope": "end-to-end"' in encoded
    assert record["attention_timing_version"] == attention.ATTENTION_TIMING_VERSION
    assert record["npu_timing_scope"] == attention.NPU_TIMING_SCOPE
    assert record["npu_timings_us"] == []


def test_baseline_worker_times_validation_failure(tmp_path, monkeypatch, capsys):
    calls = []

    def module(*_args):
        calls.append("module")

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("expected mismatch")

    def fake_measure(run_once, warmup, iterations):
        assert (warmup, iterations) == (1, 2)
        run_once()
        return [10.0, 11.0]

    monkeypatch.setattr(
        attention, "_build_baseline_modules", lambda *_args: (module,) * 4
    )
    monkeypatch.setattr(attention.np.testing, "assert_allclose", fail_validation)
    monkeypatch.setattr(attention, "measure_complete_attention", fake_measure)

    result = attention.worker_main(
        [
            "--device",
            "xdna1",
            "--implementation",
            "baseline",
            "--seq-len",
            "64",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--project-root",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert output.count(attention.VALIDATION_FAILED) == 1
    assert output.count(attention.TIMING_PREFIX) == 2
    assert len(calls) == 8


def test_timed_baseline_validation_failure_is_completed(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(attention, "DEFAULT_WORK_DIR", work_dir)
    monkeypatch.setattr(attention.base, "check_environment", lambda *_args: None)
    worker_output = "\n".join(
        [
            attention.VALIDATION_FAILED,
            f"{attention.TIMING_PREFIX}10",
            f"{attention.TIMING_PREFIX}11",
            npu_output([0.25] * 16),
        ]
    )
    monkeypatch.setattr(
        attention.base,
        "run_command",
        lambda *_args: (0, worker_output),
    )

    result = attention.main(
        [
            "run",
            "--implementation",
            "baseline",
            "--seq-len",
            "64",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    case = attention.AttentionCase("baseline", 64)
    record = json.loads(
        attention.record_path(output_dir, case).read_text(encoding="utf-8")
    )
    assert result == 0
    assert record["status"] == "failed"
    assert record["validation"] == "failed"
    assert record["timed_validation_failure"] is True
    assert record["npu_timings_us"] == [1.0, 1.0]
    assert record["timings_us"] == [10.0, 11.0]
    assert attention.is_resumable(
        attention.record_path(output_dir, case),
        attention.case_signature(case, 1, 2),
    )
    with (output_dir / "filtered_timings.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert [float(row["time_us"]) for row in rows] == [10.0, 11.0]
    assert [float(row["npu_time_us"]) for row in rows] == [1.0, 1.0]
    assert [float(row["extra_time_us"]) for row in rows] == [9.0, 10.0]


def test_infeasible_baseline_is_recorded_without_hardware(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(attention, "DEFAULT_WORK_DIR", work_dir)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("infeasible case must not access hardware")

    monkeypatch.setattr(attention.base, "check_environment", unexpected)
    monkeypatch.setattr(attention.base, "run_command", unexpected)

    result = attention.main(
        [
            "run",
            "--implementation",
            "baseline",
            "--seq-len",
            "2048",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    case = attention.AttentionCase("baseline", 2048)
    record = json.loads(
        attention.record_path(output_dir, case).read_text(encoding="utf-8")
    )
    assert record["npu_timings_us"] == []
    assert result == 0
    assert record["status"] == "infeasible"
    assert record["validation"] == "not_run"
    assert record["timings_us"] == []
    assert record["commands"] == []
    assert attention.is_resumable(
        attention.record_path(output_dir, case),
        attention.case_signature(case, 1, 2),
    )
    assert not attention.work_path(case).exists()
    assert attention.INFEASIBLE_BASELINE_REASON in attention.log_path(
        output_dir, case
    ).read_text(encoding="utf-8")
