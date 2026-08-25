# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NPU GEMM experiment runner."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

MODULE_PATH = Path(__file__).parents[1] / "gemm.py"
SPEC = importlib.util.spec_from_file_location("aie_experiments_gemm", MODULE_PATH)
gemm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gemm
SPEC.loader.exec_module(gemm)


def one_case(flow="allo", dtype="int16", M=256, N=256, K=256):
    """Return one case through the public configuration generator."""
    return gemm.generate_cases(flow, [dtype], [M], [N], [K])[0]


def test_full_sweep_counts_and_tilings():
    cases = gemm.generate_cases(
        "both",
        gemm.DEFAULT_DTYPES,
        gemm.DEFAULT_SIZES,
        gemm.DEFAULT_SIZES,
        gemm.DEFAULT_SIZES,
    )
    assert len(cases) == 384
    assert sum(case.flow == "allo" for case in cases) == 192
    assert sum(case.flow == "mlir-aie" for case in cases) == 192
    assert gemm.tiling_for("int16") == (64, 64, 64)
    assert gemm.tiling_for("bf16") == (64, 64, 64)
    assert gemm.tiling_for("int8") == (64, 128, 64)


def test_int8_n256_uses_two_columns_only():
    int8_small_n = one_case(dtype="int8", N=256)
    int8_large_n = one_case(dtype="int8", N=512)
    int16_small_n = one_case(dtype="int16", N=256)
    assert int8_small_n.npu_columns == 2
    assert int8_large_n.npu_columns == 4
    assert int16_small_n.npu_columns == 4


@pytest.mark.parametrize(
    "statement", ["total_npu_time += npu_time;", "npu_time_total += npu_time;"]
)
def test_instrument_timing_source(statement):
    source = f"void run() {{\n  {statement}\n}}\n"
    instrumented = gemm.instrument_timing_source(source)
    assert instrumented.count(gemm.TIMING_PREFIX) == 1
    assert statement in instrumented


def test_instrument_timing_source_rejects_unknown_shape():
    with pytest.raises(gemm.ExperimentError, match="found 0"):
        gemm.instrument_timing_source("void run() {}\n")


def test_parse_sample_timings():
    output = "noise\nNPU_SAMPLE_US=10\nNPU_SAMPLE_US=12.5\n"
    assert gemm.parse_sample_timings(output, 2) == [10.0, 12.5]
    with pytest.raises(gemm.ExperimentError, match="captured 2"):
        gemm.parse_sample_timings(output, 3)
    with pytest.raises(gemm.ExperimentError, match="Malformed"):
        gemm.parse_sample_timings("NPU_SAMPLE_US=bad\n", 1)
    with pytest.raises(gemm.ExperimentError, match="Invalid"):
        gemm.parse_sample_timings("NPU_SAMPLE_US=0\n", 1)


def test_tukey_filter_outlier_and_zero_iqr():
    result = gemm.tukey_filter([10.0] * 8 + [100.0])
    assert result["iqr"] == 0.0
    assert result["filtered"] == [10.0] * 8
    assert result["mask"][-1] is False

    equal_result = gemm.tukey_filter([7.0] * 5)
    assert equal_result["filtered"] == [7.0] * 5
    assert all(equal_result["mask"])


def test_mlir_command_maps_dimensions_tiles_and_dtype():
    case = one_case(flow="mlir-aie", dtype="int8", M=256, N=256, K=512)
    command = gemm.mlir_make_command(case, Path("/opt/mlir-aie"))
    assert "M=256" in command
    assert "K=512" in command
    assert "N=256" in command
    assert "m=64" in command
    assert "k=64" in command
    assert "n=128" in command
    assert "n_aie_cols=2" in command
    assert "dtype_in=i8" in command
    assert "dtype_out=i8" in command
    assert command[-1] == "build/final_256x512x256_64x64x128_2c.xclbin"


def test_resume_requires_matching_successful_sample_count(tmp_path):
    case = one_case()
    signature = gemm.case_signature(case, warmup=2, iterations=3)
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "status": "success",
                "signature": signature,
                "timings_us": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    assert gemm.is_resumable(path, signature)
    signature["iterations"] = 4
    assert not gemm.is_resumable(path, signature)


def test_process_results_writes_raw_filtered_and_summary(tmp_path):
    output_dir = tmp_path / "results"
    case = one_case()
    record = gemm.new_record(case, warmup=2, iterations=5, output_dir=output_dir)
    record.update(
        {
            "status": "success",
            "validation": "passed",
            "timings_us": [10.0, 10.0, 10.0, 10.0, 100.0],
            "elapsed_seconds": 1.25,
        }
    )
    gemm.atomic_write_json(gemm.record_path(output_dir, case), record)

    raw_path, filtered_path, summary_path = gemm.process_results(output_dir)
    with raw_path.open(encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    with filtered_path.open(encoding="utf-8") as source:
        filtered_rows = list(csv.DictReader(source))
    with summary_path.open(encoding="utf-8") as source:
        summary_rows = list(csv.DictReader(source))

    assert len(raw_rows) == 5
    assert len(filtered_rows) == 4
    assert raw_rows[-1]["is_outlier"] == "True"
    assert len(summary_rows) == 1
    assert summary_rows[0]["raw_count"] == "5"
    assert summary_rows[0]["filtered_count"] == "4"
    assert summary_rows[0]["outlier_count"] == "1"
    assert float(summary_rows[0]["filtered_mean_us"]) == 10.0


def test_safe_remove_work_only_removes_descendants(tmp_path, monkeypatch):
    work_root = tmp_path / ".work" / "gemm"
    child = work_root / "allo" / "case"
    child.mkdir(parents=True)
    (child / "artifact").write_text("data", encoding="utf-8")
    monkeypatch.setattr(gemm, "DEFAULT_WORK_DIR", work_root)
    gemm.safe_remove_work(child)
    assert not child.exists()
    with pytest.raises(gemm.ExperimentError, match="unsafe"):
        gemm.safe_remove_work(work_root)


def test_list_and_dry_run_do_not_require_hardware(capsys):
    selection = [
        "--flow",
        "both",
        "--dtype",
        "int8",
        "--M",
        "256",
        "--N",
        "256",
        "--K",
        "256",
    ]
    assert gemm.main(["list", *selection]) == 0
    assert "Total configurations: 2" in capsys.readouterr().out

    assert gemm.main(["run", *selection, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Dry run: 2 configuration(s)" in output
    assert "n_aie_cols=2" in output


def test_validation_failure_flag_is_scoped_in_dry_run(capsys):
    selection = [
        "--flow",
        "both",
        "--dtype",
        "int8",
        "bf16",
        "--M",
        "256",
        "--N",
        "256",
        "--K",
        "256",
    ]
    assert (
        gemm.main(
            [
                "run",
                *selection,
                "--benchmark-on-validation-failure",
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert output.count("--benchmark-on-validation-failure") == 1

    assert (
        gemm.main(
            [
                "run",
                "--flow",
                "mlir-aie",
                "--dtype",
                "bf16",
                "--benchmark-on-validation-failure",
                "--dry-run",
            ]
        )
        == 2
    )
    assert "requires an Allo bf16 selection" in capsys.readouterr().err


def test_run_allo_case_accepts_only_opted_in_failed_marker(tmp_path, monkeypatch):
    case = one_case(dtype="bf16")
    output = "\n".join(
        [
            gemm.ALLO_VALIDATION_FAILED,
            f"{gemm.TIMING_PREFIX}10",
            f"{gemm.TIMING_PREFIX}11",
        ]
    )
    captured_command = []

    def fake_run_command(command, _cwd, _log_file, _env):
        captured_command.extend(command)
        return 0, output

    monkeypatch.setattr(gemm, "run_command", fake_run_command)
    timings, commands, validation = gemm.run_allo_case(
        case,
        1,
        2,
        tmp_path / "work",
        tmp_path / "case.log",
        {},
        True,
    )
    assert timings == [10.0, 11.0]
    assert commands
    assert validation == "failed"
    assert "--benchmark-on-validation-failure" in captured_command

    with pytest.raises(gemm.ExperimentError, match="unexpectedly"):
        gemm.run_allo_case(
            case,
            1,
            2,
            tmp_path / "strict-work",
            tmp_path / "strict.log",
            {},
            False,
        )


def test_timed_validation_failure_is_completed_and_returns_zero(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(gemm, "DEFAULT_WORK_DIR", work_dir)
    monkeypatch.setattr(gemm, "check_environment", lambda *_args: None)

    def fake_run_allo_case(*args):
        assert args[-1] is True
        return [10.0, 10.0, 10.0], ["worker command"], "failed"

    monkeypatch.setattr(gemm, "run_allo_case", fake_run_allo_case)
    assert (
        gemm.main(
            [
                "run",
                "--flow",
                "allo",
                "--dtype",
                "bf16",
                "--M",
                "256",
                "--N",
                "256",
                "--K",
                "256",
                "--warmup",
                "1",
                "--iterations",
                "3",
                "--output-dir",
                str(output_dir),
                "--benchmark-on-validation-failure",
            ]
        )
        == 0
    )

    case = one_case(dtype="bf16")
    record = json.loads(gemm.record_path(output_dir, case).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["validation"] == "failed"
    assert record["timed_validation_failure"] is True
    assert record["timings_us"] == [10.0, 10.0, 10.0]
    assert "--benchmark-on-validation-failure" in record["error"]
    assert not gemm.work_path(case).exists()


def test_unexpected_failure_remains_hard_and_untimed(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    work_dir = tmp_path / "work"
    monkeypatch.setattr(gemm, "DEFAULT_WORK_DIR", work_dir)
    monkeypatch.setattr(gemm, "check_environment", lambda *_args: None)

    def fail_run(*_args):
        raise gemm.ExperimentError("compile failed")

    monkeypatch.setattr(gemm, "run_allo_case", fail_run)
    assert (
        gemm.main(
            [
                "run",
                "--flow",
                "allo",
                "--dtype",
                "bf16",
                "--M",
                "256",
                "--N",
                "256",
                "--K",
                "256",
                "--iterations",
                "3",
                "--output-dir",
                str(output_dir),
                "--benchmark-on-validation-failure",
            ]
        )
        == 1
    )
    case = one_case(dtype="bf16")
    record = json.loads(gemm.record_path(output_dir, case).read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["timed_validation_failure"] is False
    assert record["timings_us"] == []
    assert gemm.work_path(case).exists()


def test_timed_validation_failure_resume_requires_matching_opt_in_and_samples(
    tmp_path,
):
    case = one_case(dtype="bf16")
    signature = gemm.case_signature(case, 2, 3, True)
    path = tmp_path / "case.json"
    path.write_text(
        json.dumps(
            {
                "flow": case.flow,
                "dtype": case.dtype,
                "status": "failed",
                "validation": "failed",
                "timed_validation_failure": True,
                "signature": signature,
                "timings_us": [1.0, 2.0, 3.0],
            }
        ),
        encoding="utf-8",
    )
    assert gemm.is_resumable(path, signature)
    assert not gemm.is_resumable(path, gemm.case_signature(case, 2, 3))

    record = json.loads(path.read_text(encoding="utf-8"))
    record["timings_us"].pop()
    path.write_text(json.dumps(record), encoding="utf-8")
    assert not gemm.is_resumable(path, signature)


def test_process_results_filters_only_marked_failed_timings(tmp_path):
    output_dir = tmp_path / "results"
    timed_case = one_case(dtype="bf16")
    timed_record = gemm.new_record(
        timed_case,
        warmup=2,
        iterations=5,
        output_dir=output_dir,
        benchmark_on_validation_failure=True,
    )
    timed_record.update(
        {
            "status": "failed",
            "validation": "failed",
            "timed_validation_failure": True,
            "timings_us": [10.0, 10.0, 10.0, 10.0, 100.0],
        }
    )
    gemm.atomic_write_json(gemm.record_path(output_dir, timed_case), timed_record)

    ordinary_case = one_case(dtype="int16")
    ordinary_record = gemm.new_record(
        ordinary_case, warmup=2, iterations=2, output_dir=output_dir
    )
    ordinary_record.update(
        {
            "status": "failed",
            "validation": "failed",
            "timings_us": [20.0, 21.0],
            "error": "runtime failed after samples",
        }
    )
    gemm.atomic_write_json(gemm.record_path(output_dir, ordinary_case), ordinary_record)

    raw_path, filtered_path, summary_path = gemm.process_results(output_dir)
    with raw_path.open(encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source))
    with filtered_path.open(encoding="utf-8") as source:
        filtered_rows = list(csv.DictReader(source))
    with summary_path.open(encoding="utf-8") as source:
        summaries = {row["case_id"]: row for row in csv.DictReader(source)}

    assert len(raw_rows) == 7
    assert len(filtered_rows) == 4
    assert all(row["timed_validation_failure"] == "True" for row in filtered_rows)
    timed_summary = summaries[timed_case.case_id]
    assert timed_summary["status"] == "failed"
    assert timed_summary["timed_validation_failure"] == "True"
    assert timed_summary["filtered_count"] == "4"
    assert float(timed_summary["filtered_mean_us"]) == 10.0
    ordinary_summary = summaries[ordinary_case.case_id]
    assert ordinary_summary["timed_validation_failure"] == "False"
    assert ordinary_summary["filtered_count"] == "0"
    assert ordinary_summary["filtered_mean_us"] == ""


def test_allo_worker_device_column_override_is_optional(tmp_path):
    case = one_case(dtype="bf16")
    default_command = gemm.allo_worker_command(case, 1, 2, tmp_path / "default.prj")
    assert "--device-columns" not in default_command
    assert "--rows" not in default_command

    overridden_command = gemm.allo_worker_command(
        case,
        1,
        2,
        tmp_path / "overridden.prj",
        benchmark_on_validation_failure=True,
        device_columns=4,
        mapping_rows=2,
    )
    assert overridden_command[overridden_command.index("--device-columns") + 1] == "4"
    assert overridden_command[overridden_command.index("--rows") + 1] == "2"
    assert "--benchmark-on-validation-failure" in overridden_command
