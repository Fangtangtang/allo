# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the partial-NPU bf16 GEMM experiment runner."""

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest

EXPERIMENT_DIR = Path(__file__).parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
MODULE_PATH = EXPERIMENT_DIR / "gemm_partial_npu.py"
SPEC = importlib.util.spec_from_file_location(
    "aie_experiments_gemm_partial_npu", MODULE_PATH
)
partial = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = partial
SPEC.loader.exec_module(partial)


def one_shape_cases(columns=2, variants=partial.VARIANT_ORDER):
    """Return physical cases for one matrix shape."""
    return partial.generate_cases(variants, [columns], [256], [256], [256])


def command_value(command, flag):
    """Return the argument immediately following a command flag."""
    return command[command.index(flag) + 1]


def test_default_expansion_has_512_physical_and_576_logical_cases():
    cases = partial.generate_cases(
        partial.VARIANT_ORDER,
        partial.DEFAULT_COLUMNS,
        partial.base.DEFAULT_SIZES,
        partial.base.DEFAULT_SIZES,
        partial.base.DEFAULT_SIZES,
    )
    assert len(cases) == 512
    assert partial.logical_case_count(cases) == 576
    assert sum(case.variant == "manual" for case in cases) == 192
    assert sum(case.variant == "compiled" for case in cases) == 192
    assert sum(case.variant == "compiled-full-io" for case in cases) == 128
    for columns in partial.DEFAULT_COLUMNS:
        selected = [case for case in cases if case.compute_columns == columns]
        assert partial.logical_case_count(selected) == 192


def test_backend_commands_separate_mapping_and_device_columns():
    cases = {case.variant: case for case in one_shape_cases(columns=2)}

    manual = partial.planned_command(
        cases["manual"], 3, 7, Path("/tmp/manual"), Path("/opt/mlir-aie")
    )
    assert "n_aie_cols=2" in manual
    assert "dtype_in=bf16" in manual
    assert "dtype_out=bf16" in manual

    compiled = partial.allo_command(cases["compiled"], 3, 7, Path("/tmp/compiled.prj"))
    assert command_value(compiled, "--columns") == "2"
    assert "--device-columns" not in compiled
    assert "--benchmark-on-validation-failure" in compiled

    full_io = partial.allo_command(
        cases["compiled-full-io"], 3, 7, Path("/tmp/full-io.prj")
    )
    assert command_value(full_io, "--columns") == "2"
    assert command_value(full_io, "--device-columns") == "4"
    assert "--benchmark-on-validation-failure" in full_io


def test_four_column_allo_configuration_is_deduplicated_and_aliased():
    cases = one_shape_cases(columns=4)
    assert [case.variant for case in cases] == ["manual", "compiled"]
    assert partial.logical_case_count(cases) == 3
    compiled = cases[1]
    assert compiled.compute_columns == 4
    assert compiled.device_columns == 4
    assert compiled.plot_series == ("compiled", "compiled-full-io")
    assert "--device-columns" not in partial.allo_command(
        compiled, 1, 2, Path("/tmp/compiled.prj")
    )

    full_io_only = one_shape_cases(columns=4, variants=["compiled-full-io"])
    assert len(full_io_only) == 1
    assert full_io_only[0].variant == "compiled"
    assert full_io_only[0].plot_series == ("compiled", "compiled-full-io")


def test_list_and_dry_run_require_no_hardware(capsys):
    selection = [
        "--columns",
        "2",
        "--M",
        "256",
        "--N",
        "256",
        "--K",
        "256",
    ]
    assert partial.main(["list", *selection]) == 0
    output = capsys.readouterr().out
    assert "Total physical configurations: 3" in output
    assert "Total plotted series points: 3" in output

    assert partial.main(["run", *selection, "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "Dry run: 3 physical configuration(s), 3 plotted series point(s)" in output
    assert "n_aie_cols=2" in output
    assert output.count("--benchmark-on-validation-failure") == 2
    assert output.count("--device-columns 4") == 1


def test_timed_allo_failure_is_filtered_resumable_and_returns_zero(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "results"
    work_root = tmp_path / ".work" / "gemm"
    monkeypatch.setattr(partial.base, "DEFAULT_WORK_DIR", work_root)
    monkeypatch.setattr(partial, "DEFAULT_WORK_DIR", work_root / "partial-npu")
    monkeypatch.setattr(partial.base, "check_environment", lambda *_args: None)

    calls = []

    def fake_run(case, _args, _case_work, _case_log, _environment):
        calls.append(case)
        return [10.0, 10.0, 10.0, 10.0, 100.0], ["worker command"], "failed"

    monkeypatch.setattr(partial, "run_physical_case", fake_run)
    selection = [
        "--variant",
        "compiled",
        "--columns",
        "1",
        "--M",
        "256",
        "--N",
        "256",
        "--K",
        "256",
        "--warmup",
        "2",
        "--iterations",
        "5",
        "--output-dir",
        str(output_dir),
    ]
    assert partial.main(["run", *selection]) == 0
    assert len(calls) == 1

    case = one_shape_cases(columns=1, variants=["compiled"])[0]
    result_path = partial.record_path(output_dir, case)
    record = json.loads(result_path.read_text(encoding="utf-8"))
    assert record["variant"] == "compiled"
    assert record["compute_columns"] == 1
    assert record["device_columns"] == 1
    assert record["mapping_columns"] == 2
    assert record["mapping_rows"] == 2
    assert record["status"] == "failed"
    assert record["validation"] == "failed"
    assert record["timed_validation_failure"] is True
    assert record["timings_us"] == [10.0, 10.0, 10.0, 10.0, 100.0]
    assert record["signature"]["benchmark_on_validation_failure"] is True
    assert partial.base.is_resumable(
        result_path, partial.case_signature(case, warmup=2, iterations=5)
    )
    assert not partial.work_path(case).exists()

    with (output_dir / "filtered_timings.csv").open(encoding="utf-8") as source:
        filtered_rows = list(csv.DictReader(source))
    with (output_dir / "summary.csv").open(encoding="utf-8") as source:
        summary_rows = list(csv.DictReader(source))
    assert len(filtered_rows) == 4
    assert all(row["timed_validation_failure"] == "True" for row in filtered_rows)
    assert summary_rows[0]["filtered_count"] == "4"
    assert summary_rows[0]["plot_series"] == "compiled"

    assert partial.main(["run", *selection]) == 0
    assert len(calls) == 1
    assert partial.main(["process", "--output-dir", str(output_dir)]) == 0


def test_unexpected_failure_is_hard_untimed_and_keeps_build(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"
    work_root = tmp_path / ".work" / "gemm"
    monkeypatch.setattr(partial.base, "DEFAULT_WORK_DIR", work_root)
    monkeypatch.setattr(partial, "DEFAULT_WORK_DIR", work_root / "partial-npu")
    monkeypatch.setattr(partial.base, "check_environment", lambda *_args: None)

    def fail_run(*_args):
        raise partial.base.ExperimentError("compile failed")

    monkeypatch.setattr(partial, "run_physical_case", fail_run)
    assert (
        partial.main(
            [
                "run",
                "--variant",
                "compiled-full-io",
                "--columns",
                "2",
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
            ]
        )
        == 1
    )

    case = one_shape_cases(columns=2, variants=["compiled-full-io"])[0]
    record = json.loads(
        partial.record_path(output_dir, case).read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["validation"] == "failed"
    assert record["timed_validation_failure"] is False
    assert record["timings_us"] == []
    assert record["error"] == "compile failed"
    assert partial.work_path(case).exists()


@pytest.mark.parametrize(
    ("option", "value"), [("--warmup", "-1"), ("--iterations", "0")]
)
def test_invalid_timing_arguments_are_rejected(option, value, capsys):
    assert (
        partial.main(
            [
                "run",
                "--variant",
                "compiled",
                "--columns",
                "1",
                "--M",
                "256",
                "--N",
                "256",
                "--K",
                "256",
                option,
                value,
                "--dry-run",
            ]
        )
        == 1
    )
    assert "must be" in capsys.readouterr().err


def test_one_by_four_compiled_uses_two_by_two_mapping_on_one_column_device():
    cases = {case.variant: case for case in one_shape_cases(columns=1)}
    manual = cases["manual"]
    compiled = cases["compiled"]
    full_io = cases["compiled-full-io"]

    assert (manual.mapping_rows, manual.mapping_columns, manual.device_columns) == (
        4,
        1,
        1,
    )
    assert (
        compiled.mapping_rows,
        compiled.mapping_columns,
        compiled.device_columns,
    ) == (2, 2, 1)
    assert (full_io.mapping_rows, full_io.mapping_columns, full_io.device_columns) == (
        4,
        1,
        4,
    )

    command = partial.allo_command(compiled, 3, 7, Path("/tmp/compiled.prj"))
    assert command_value(command, "--columns") == "2"
    assert command_value(command, "--rows") == "2"
    assert command_value(command, "--device-columns") == "1"


def test_one_by_four_compiled_uses_two_by_two_mapping_on_one_column_device():
    cases = {case.variant: case for case in one_shape_cases(columns=1)}
    manual = cases["manual"]
    compiled = cases["compiled"]
    full_io = cases["compiled-full-io"]

    assert (manual.mapping_rows, manual.mapping_columns, manual.device_columns) == (
        4,
        1,
        1,
    )
    assert (
        compiled.mapping_rows,
        compiled.mapping_columns,
        compiled.device_columns,
    ) == (2, 2, 1)
    assert (full_io.mapping_rows, full_io.mapping_columns, full_io.device_columns) == (
        4,
        1,
        4,
    )

    command = partial.allo_command(compiled, 3, 7, Path("/tmp/compiled.prj"))
    assert command_value(command, "--columns") == "2"
    assert command_value(command, "--rows") == "2"
    assert command_value(command, "--device-columns") == "1"
