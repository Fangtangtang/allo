#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and process GEMM performance experiments on Ryzen AI NPUs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "gemm"
DEFAULT_WORK_DIR = EXPERIMENT_DIR / ".work" / "gemm"
DEFAULT_MLIR_AIE_ROOT = Path("/ryzers/mlir-aie")
DEFAULT_SIZES = (256, 512, 1024, 2048)
DEFAULT_DTYPES = ("int16", "int8", "bf16")
FLOW_ORDER = ("allo", "mlir-aie")
SCHEMA_VERSION = 1
TIMING_PREFIX = "NPU_SAMPLE_US="
ALLO_VALIDATION_PASSED = "ALLO_VALIDATION=PASSED"
ALLO_VALIDATION_FAILED = "ALLO_VALIDATION=FAILED"


class ExperimentError(RuntimeError):
    """An expected experiment setup or execution failure."""


@dataclass(frozen=True)
class GemmCase:
    """One concrete GEMM configuration."""

    flow: str
    dtype: str
    M: int
    N: int
    K: int
    m: int
    n: int
    k: int
    npu_columns: int

    @property
    def case_id(self) -> str:
        """Return a stable identifier for result and work paths."""
        return (
            f"{self.flow}_{self.dtype}_M{self.M}_N{self.N}_K{self.K}_"
            f"m{self.m}_n{self.n}_k{self.k}_{self.npu_columns}col"
        )


def utc_now() -> str:
    """Return an ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def tiling_for(dtype: str) -> tuple[int, int, int]:
    """Return (m, n, k) for one supported datatype."""
    if dtype == "int8":
        return 64, 128, 64
    if dtype in {"int16", "bf16"}:
        return 64, 64, 64
    raise ValueError(f"Unsupported datatype: {dtype}")


def npu_columns_for(N: int, n: int) -> int:
    """Select the largest valid NPU1 column count for the N tiling."""
    for columns in (4, 2, 1):
        if N % (n * columns) == 0:
            return columns
    raise ValueError(f"N={N} cannot be distributed with tile n={n} on NPU1")


def expand_flows(flow: str) -> tuple[str, ...]:
    """Expand the public 'both' flow selector."""
    if flow == "both":
        return FLOW_ORDER
    if flow in FLOW_ORDER:
        return (flow,)
    raise ValueError(f"Unsupported flow: {flow}")


def unique(values: Iterable[object]) -> list[object]:
    """Remove duplicates while preserving command-line order."""
    return list(dict.fromkeys(values))


def generate_cases(
    flow: str,
    dtypes: Sequence[str],
    matrix_ms: Sequence[int],
    matrix_ns: Sequence[int],
    matrix_ks: Sequence[int],
) -> list[GemmCase]:
    """Generate the requested Cartesian product in deterministic order."""
    cases = []
    for selected_flow in expand_flows(flow):
        for dtype in unique(dtypes):
            m, n, k = tiling_for(str(dtype))
            for M, N, K in itertools.product(
                unique(matrix_ms), unique(matrix_ns), unique(matrix_ks)
            ):
                cases.append(
                    GemmCase(
                        selected_flow,
                        str(dtype),
                        int(M),
                        int(N),
                        int(K),
                        m,
                        n,
                        k,
                        npu_columns_for(int(N), n),
                    )
                )
    return cases


def case_signature(
    case: GemmCase,
    warmup: int,
    iterations: int,
    benchmark_on_validation_failure: bool = False,
) -> dict:
    """Return the fields that determine whether a result is resumable."""
    signature = {
        "schema_version": SCHEMA_VERSION,
        **asdict(case),
        "warmup": warmup,
        "iterations": iterations,
        "target": "npu1",
        "output_dtype": case.dtype,
    }

    if benchmark_on_validation_failure:
        signature["benchmark_on_validation_failure"] = True
    return signature


def is_timed_validation_failure(record: dict) -> bool:
    """Return whether a record is the expected timed Allo bf16 failure."""
    return (
        record.get("flow") == "allo"
        and record.get("dtype") == "bf16"
        and record.get("status") == "failed"
        and record.get("validation") == "failed"
        and record.get("timed_validation_failure") is True
    )


def instrument_timing_source(source: str) -> str:
    """Add one machine-readable print to an Allo or mlir-aie host loop."""
    pattern = re.compile(
        r"^(?P<indent>\s*)(?P<statement>"
        r"(?:total_npu_time|npu_time_total) \+= npu_time;)\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ExperimentError(
            "Expected exactly one measured-time accumulation statement in the "
            f"host source, found {len(matches)}"
        )

    def replacement(match: re.Match) -> str:
        indent = match.group("indent")
        statement = match.group("statement")
        return (
            f'{indent}std::cout << "{TIMING_PREFIX}" << npu_time '
            f"<< std::endl;\n{indent}{statement}"
        )

    return pattern.sub(replacement, source, count=1)


def parse_sample_timings(output: str, expected_count: int) -> list[float]:
    """Parse and validate machine-readable per-iteration device timings."""
    timings = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith(TIMING_PREFIX):
            continue
        value_text = stripped[len(TIMING_PREFIX) :]
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ExperimentError(f"Malformed timing line: {stripped}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ExperimentError(f"Invalid NPU timing value: {value_text}")
        timings.append(value)
    if len(timings) != expected_count:
        raise ExperimentError(
            f"Expected {expected_count} NPU samples, captured {len(timings)}"
        )
    return timings


def tukey_filter(values: Sequence[float]) -> dict:
    """Return Tukey-IQR bounds, mask, and retained values."""
    if not values:
        return {
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "mask": [],
            "filtered": [],
        }
    samples = np.asarray(values, dtype=np.float64)
    try:
        q1, q3 = np.percentile(samples, [25, 75], method="linear")
    except TypeError:  # NumPy < 1.22
        q1, q3 = np.percentile(samples, [25, 75], interpolation="linear")
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask_array = (samples >= lower_bound) & (samples <= upper_bound)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "mask": [bool(value) for value in mask_array],
        "filtered": [float(value) for value in samples[mask_array]],
    }


def atomic_write_json(path: Path, value: dict) -> None:
    """Write JSON without exposing a partially written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict]) -> None:
    """Write a complete CSV file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_records(output_dir: Path) -> list[dict]:
    """Load all per-case records from an output directory."""
    records = []
    for path in sorted((output_dir / "cases").glob("**/*.json")):
        with path.open("r", encoding="utf-8") as source:
            record = json.load(source)
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ExperimentError(f"Unsupported result schema in {path}")
        records.append(record)
    return sorted(
        records,
        key=lambda item: (
            FLOW_ORDER.index(item["flow"]),
            DEFAULT_DTYPES.index(item["dtype"]),
            item["M"],
            item["N"],
            item["K"],
        ),
    )


def gflops_for(case_record: dict, time_us: float) -> float:
    """Compute GEMM GFLOP/s from one execution time."""
    operations = 2.0 * case_record["M"] * case_record["N"] * case_record["K"]
    return operations / (time_us * 1000.0)


def describe_timings(record: dict, values: Sequence[float], prefix: str) -> dict:
    """Build summary columns for a timing sequence."""
    empty = {
        f"{prefix}_mean_us": "",
        f"{prefix}_median_us": "",
        f"{prefix}_min_us": "",
        f"{prefix}_max_us": "",
        f"{prefix}_std_us": "",
        f"{prefix}_gflops": "",
    }
    if not values:
        return empty
    samples = [float(value) for value in values]
    mean_us = statistics.fmean(samples)
    return {
        f"{prefix}_mean_us": mean_us,
        f"{prefix}_median_us": statistics.median(samples),
        f"{prefix}_min_us": min(samples),
        f"{prefix}_max_us": max(samples),
        f"{prefix}_std_us": statistics.pstdev(samples),
        f"{prefix}_gflops": gflops_for(record, mean_us),
    }


def process_results(output_dir: Path) -> tuple[Path, Path, Path]:
    """Regenerate raw, filtered, and summary CSV files."""
    output_dir = output_dir.resolve()
    records = load_records(output_dir)
    timing_fields = [
        "case_id",
        "flow",
        "status",
        "validation",
        "timed_validation_failure",
        "dtype",
        "M",
        "N",
        "K",
        "m",
        "n",
        "k",
        "npu_columns",
        "sample_index",
        "time_us",
        "gflops",
        "is_outlier",
    ]
    summary_fields = [
        "case_id",
        "flow",
        "status",
        "validation",
        "timed_validation_failure",
        "dtype",
        "M",
        "N",
        "K",
        "m",
        "n",
        "k",
        "npu_columns",
        "warmup",
        "iterations",
        "raw_count",
        "filtered_count",
        "outlier_count",
        "q1_us",
        "q3_us",
        "iqr_us",
        "lower_bound_us",
        "upper_bound_us",
        "raw_mean_us",
        "raw_median_us",
        "raw_min_us",
        "raw_max_us",
        "raw_std_us",
        "raw_gflops",
        "filtered_mean_us",
        "filtered_median_us",
        "filtered_min_us",
        "filtered_max_us",
        "filtered_std_us",
        "filtered_gflops",
        "elapsed_seconds",
        "error",
        "log_path",
    ]
    raw_rows = []
    filtered_rows = []
    summary_rows = []
    for record in records:
        timings = [float(value) for value in record.get("timings_us", [])]
        filter_result = tukey_filter(timings)
        mask = filter_result["mask"]
        completed = record.get("status") == "success" or is_timed_validation_failure(
            record
        )
        for index, value in enumerate(timings):
            row = {
                field: record[field]
                for field in timing_fields
                if field in record
                and field not in {"sample_index", "time_us", "gflops", "is_outlier"}
            }
            row.update(
                {
                    "sample_index": index,
                    "time_us": value,
                    "gflops": gflops_for(record, value),
                    "is_outlier": not mask[index],
                }
            )
            raw_rows.append(row)
            if completed and mask[index]:
                filtered_rows.append(row.copy())

        filtered = filter_result["filtered"] if completed else []
        summary = {
            field: record.get(field, "")
            for field in summary_fields
            if field
            not in {
                "raw_count",
                "filtered_count",
                "outlier_count",
                "q1_us",
                "q3_us",
                "iqr_us",
                "lower_bound_us",
                "upper_bound_us",
            }
        }
        summary.update(
            {
                "raw_count": len(timings),
                "filtered_count": len(filtered),
                "outlier_count": len(timings) - len(filtered) if completed else "",
                "q1_us": filter_result["q1"] if timings else "",
                "q3_us": filter_result["q3"] if timings else "",
                "iqr_us": filter_result["iqr"] if timings else "",
                "lower_bound_us": filter_result["lower_bound"] if timings else "",
                "upper_bound_us": filter_result["upper_bound"] if timings else "",
            }
        )
        summary.update(describe_timings(record, timings, "raw"))
        summary.update(describe_timings(record, filtered, "filtered"))
        summary_rows.append(summary)

    raw_path = output_dir / "raw_timings.csv"
    filtered_path = output_dir / "filtered_timings.csv"
    summary_path = output_dir / "summary.csv"
    atomic_write_csv(raw_path, timing_fields, raw_rows)
    atomic_write_csv(filtered_path, timing_fields, filtered_rows)
    atomic_write_csv(summary_path, summary_fields, summary_rows)
    return raw_path, filtered_path, summary_path


def render_command(command: Sequence[object]) -> str:
    """Render an argv list for logs and dry runs."""
    return shlex.join(str(value) for value in command)


def run_command(
    command: Sequence[object],
    cwd: Path,
    log_file: Path,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command while teeing combined output to the console and case log."""
    argv = [str(value) for value in command]
    rendered = render_command(argv)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    output_lines = []
    with log_file.open("a", encoding="utf-8") as log:
        heading = f"\n$ (cd {cwd} && {rendered})\n"
        print(heading, end="", flush=True)
        log.write(heading)
        log.flush()
        with subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                output_lines.append(line)
            return_code = process.wait()
        footer = f"[exit code: {return_code}]\n"
        print(footer, end="", flush=True)
        log.write(footer)
    return return_code, "".join(output_lines)


def record_path(output_dir: Path, case: GemmCase) -> Path:
    """Return the JSON record path for a case."""
    return output_dir / "cases" / case.flow / case.dtype / f"{case.case_id}.json"


def log_path(output_dir: Path, case: GemmCase) -> Path:
    """Return the full build and execution log path for a case."""
    return output_dir / "logs" / case.flow / case.dtype / f"{case.case_id}.log"


def work_path(case: GemmCase) -> Path:
    """Return the isolated work directory for a case."""
    return DEFAULT_WORK_DIR / case.flow / case.case_id


def safe_remove_work(path: Path) -> None:
    """Remove only a validated descendant of the experiment work root."""
    resolved_root = DEFAULT_WORK_DIR.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ExperimentError(f"Refusing to remove unsafe work path: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def is_resumable(record_file: Path, signature: dict) -> bool:
    """Check whether an existing completed result exactly matches a run."""
    if not record_file.is_file():
        return False
    try:
        with record_file.open("r", encoding="utf-8") as source:
            record = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("signature") != signature or len(
        record.get("timings_us", [])
    ) != signature.get("iterations"):
        return False
    return record.get("status") == "success" or is_timed_validation_failure(record)


def allo_worker_command(
    case: GemmCase,
    warmup: int,
    iterations: int,
    project: Path,
    benchmark_on_validation_failure: bool = False,
    device_columns: int | None = None,
    mapping_rows: int | None = None,
) -> list[str]:
    """Build the internal Allo worker command."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_allo-worker",
        "--dtype",
        case.dtype,
        "--M",
        str(case.M),
        "--N",
        str(case.N),
        "--K",
        str(case.K),
        "--m",
        str(case.m),
        "--n",
        str(case.n),
        "--k",
        str(case.k),
        "--columns",
        str(case.npu_columns),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--project",
        str(project),
    ]

    if benchmark_on_validation_failure:
        command.append("--benchmark-on-validation-failure")
    if device_columns is not None:
        command.extend(["--device-columns", str(device_columns)])
    if mapping_rows is not None:
        command.extend(["--rows", str(mapping_rows)])
    return command


def mlir_target_suffix(case: GemmCase) -> str:
    """Return the target suffix used by the upstream Makefile."""
    return (
        f"{case.M}x{case.K}x{case.N}_{case.m}x{case.k}x{case.n}_" f"{case.npu_columns}c"
    )


def mlir_make_command(case: GemmCase, mlir_aie_root: Path) -> list[str]:
    """Build the exact upstream xclbin target for a case."""
    makefile = (
        mlir_aie_root
        / "programming_examples/basic/matrix_multiplication/whole_array/Makefile"
    )
    target = f"build/final_{mlir_target_suffix(case)}.xclbin"
    dtype = {"int16": "i16", "int8": "i8", "bf16": "bf16"}[case.dtype]
    return [
        "make",
        "-f",
        str(makefile),
        f"M={case.M}",
        f"K={case.K}",
        f"N={case.N}",
        f"m={case.m}",
        f"k={case.k}",
        f"n={case.n}",
        f"n_aie_cols={case.npu_columns}",
        f"dtype_in={dtype}",
        f"dtype_out={dtype}",
        "b_col_maj=0",
        "devicename=npu",
        target,
    ]


def mlir_host_types(dtype: str) -> tuple[str, str]:
    """Return the C++ output and reference-accumulator types."""
    mapping = {
        "int16": ("int16_t", "int16_t"),
        "int8": ("int8_t", "int8_t"),
        "bf16": ("std::bfloat16_t", "float"),
    }
    return mapping[dtype]


def ensure_mlir_host(
    case: GemmCase,
    mlir_aie_root: Path,
    log_file: Path,
    commands: list[str],
    env: dict[str, str],
) -> Path:
    """Build a datatype-specific instrumented copy of the upstream host."""
    matrix_root = (
        mlir_aie_root / "programming_examples/basic/matrix_multiplication"
    ).resolve()
    upstream_source = matrix_root / "test.cpp"
    runtime_test_lib = (mlir_aie_root / "runtime_lib/test_lib").resolve()
    source_text = instrument_timing_source(upstream_source.read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    host_root = DEFAULT_WORK_DIR / "mlir-aie-host" / case.dtype
    host_source = host_root / "test.cpp"
    executable = host_root / "whole_array_raw_timing"
    manifest_path = host_root / "manifest.json"
    output_type, accumulator_type = mlir_host_types(case.dtype)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_hash": source_hash,
        "mlir_aie_root": str(mlir_aie_root.resolve()),
        "dtype": case.dtype,
        "output_type": output_type,
        "accumulator_type": accumulator_type,
    }
    if executable.is_file() and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as source:
            previous_manifest = json.load(source)
        if previous_manifest == manifest:
            return executable

    host_root.mkdir(parents=True, exist_ok=True)
    host_source.write_text(source_text, encoding="utf-8")
    command = [
        "g++-13",
        "-std=c++23",
        "-O2",
        "-ggdb",
        "-DDISABLE_ABI_CHECK=1",
        f"-DDTYPE_IN={output_type}",
        f"-DDTYPE_OUT={output_type}",
        f"-DDTYPE_ACC={accumulator_type}",
        str(host_source),
        str(runtime_test_lib / "test_utils.cpp"),
        "-I",
        str(matrix_root),
        "-I",
        str(runtime_test_lib),
        "-I",
        "/opt/xilinx/xrt/include",
        "-L",
        "/opt/xilinx/xrt/lib",
        "-Wl,-rpath,/opt/xilinx/xrt/lib",
        "-lxrt_coreutil",
        "-lboost_program_options",
        "-lboost_filesystem",
        "-o",
        str(executable),
    ]
    commands.append(render_command(command))
    return_code, _ = run_command(command, host_root, log_file, env)
    if return_code != 0:
        raise ExperimentError("Failed to build the instrumented mlir-aie host")
    atomic_write_json(manifest_path, manifest)
    return executable


def mlir_host_command(
    executable: Path,
    case: GemmCase,
    case_work: Path,
    verify: bool,
    warmup: int,
    iterations: int,
) -> list[str]:
    """Build an invocation of the mlir-aie host executable."""
    suffix = mlir_target_suffix(case)
    return [
        str(executable),
        "-x",
        str(case_work / f"build/final_{suffix}.xclbin"),
        "-i",
        str(case_work / f"build/insts_{suffix}.txt"),
        "-k",
        "MLIR_AIE",
        "-M",
        str(case.M),
        "-K",
        str(case.K),
        "-N",
        str(case.N),
        "--b_col_maj",
        "0",
        "--verify",
        str(verify).lower(),
        "--warmup",
        str(warmup),
        "--iters",
        str(iterations),
        "-v",
        "1" if verify else "0",
    ]


def run_allo_case(
    case: GemmCase,
    warmup: int,
    iterations: int,
    case_work: Path,
    log_file: Path,
    env: dict[str, str],
    benchmark_on_validation_failure: bool = False,
    device_columns: int | None = None,
    mapping_rows: int | None = None,
) -> tuple[list[float], list[str], str]:
    """Run one Allo build, validation, and benchmark in a child process."""
    project = case_work / "allo.prj"
    command = allo_worker_command(
        case,
        warmup,
        iterations,
        project,
        benchmark_on_validation_failure,
        device_columns,
        mapping_rows,
    )
    commands = [render_command(command)]
    return_code, output = run_command(command, REPO_ROOT, log_file, env)
    if return_code != 0:
        raise ExperimentError("Allo worker failed; see the case log")
    reported = [
        validation
        for marker, validation in (
            (ALLO_VALIDATION_PASSED, "passed"),
            (ALLO_VALIDATION_FAILED, "failed"),
        )
        if marker in output
    ]
    if len(reported) != 1:
        raise ExperimentError(
            "Allo worker did not report exactly one validation result"
        )
    validation = reported[0]
    if validation == "failed" and not benchmark_on_validation_failure:
        raise ExperimentError("Allo worker reported failed validation unexpectedly")
    timings = parse_sample_timings(output, iterations)
    return timings, commands, validation


def run_mlir_aie_case(
    case: GemmCase,
    warmup: int,
    iterations: int,
    case_work: Path,
    log_file: Path,
    env: dict[str, str],
    mlir_aie_root: Path,
) -> tuple[list[float], list[str]]:
    """Run one upstream mlir-aie build, validation, and benchmark."""
    commands: list[str] = []
    executable = ensure_mlir_host(case, mlir_aie_root, log_file, commands, env)
    make_command = mlir_make_command(case, mlir_aie_root)
    commands.append(render_command(make_command))
    return_code, _ = run_command(make_command, case_work, log_file, env)
    if return_code != 0:
        raise ExperimentError("mlir-aie xclbin build failed; see the case log")

    validation_command = mlir_host_command(executable, case, case_work, True, 0, 1)
    commands.append(render_command(validation_command))
    return_code, validation_output = run_command(
        validation_command, case_work, log_file, env
    )
    if return_code != 0 or "PASS!" not in validation_output:
        raise ExperimentError("mlir-aie correctness validation failed")

    benchmark_command = mlir_host_command(
        executable, case, case_work, False, warmup, iterations
    )
    commands.append(render_command(benchmark_command))
    return_code, benchmark_output = run_command(
        benchmark_command, case_work, log_file, env
    )
    if return_code != 0:
        raise ExperimentError("mlir-aie benchmark failed; see the case log")
    return parse_sample_timings(benchmark_output, iterations), commands


def check_environment(flows: Sequence[str], mlir_aie_root: Path) -> None:
    """Fail early when required hardware or toolchain components are absent."""
    missing = []
    if not Path("/dev/accel/accel0").exists():
        missing.append("/dev/accel/accel0")
    for variable in ("MLIR_AIE_INSTALL_DIR", "RUNTIME_LIB_DIR", "PEANO_INSTALL_DIR"):
        if not os.environ.get(variable):
            missing.append(f"environment variable {variable}")
    if "mlir-aie" in flows:
        required_paths = [
            mlir_aie_root
            / "programming_examples/basic/matrix_multiplication/whole_array/Makefile",
            mlir_aie_root / "programming_examples/basic/matrix_multiplication/test.cpp",
            mlir_aie_root / "runtime_lib/test_lib/test_utils.cpp",
        ]
        missing.extend(str(path) for path in required_paths if not path.is_file())
        for program in ("make", "g++-13", "aiecc.py"):
            if shutil.which(program) is None:
                missing.append(f"executable {program}")
    if missing:
        formatted = "\n  - ".join(missing)
        raise ExperimentError(f"Missing NPU experiment requirements:\n  - {formatted}")
    if os.environ.get("CONDA_DEFAULT_ENV") != "allo":
        print(
            "WARNING: CONDA_DEFAULT_ENV is not 'allo'; follow the README setup "
            "before running hardware experiments.",
            file=sys.stderr,
        )


def new_record(
    case: GemmCase,
    warmup: int,
    iterations: int,
    output_dir: Path,
    benchmark_on_validation_failure: bool = False,
) -> dict:
    """Create a running per-case result record."""
    case_log = log_path(output_dir, case)
    try:
        stored_log_path = str(case_log.relative_to(output_dir))
    except ValueError:
        stored_log_path = str(case_log)
    return {
        "schema_version": SCHEMA_VERSION,
        **asdict(case),
        "case_id": case.case_id,
        "target": "npu1",
        "output_dtype": case.dtype,
        "warmup": warmup,
        "iterations": iterations,
        "signature": case_signature(
            case, warmup, iterations, benchmark_on_validation_failure
        ),
        "status": "running",
        "validation": "not_run",
        "benchmark_on_validation_failure": benchmark_on_validation_failure,
        "timed_validation_failure": False,
        "timings_us": [],
        "commands": [],
        "started_at": utc_now(),
        "finished_at": None,
        "elapsed_seconds": None,
        "error": "",
        "log_path": stored_log_path,
    }


def benchmark_on_validation_failure_for_case(case: GemmCase, enabled: bool) -> bool:
    """Return whether validation-failure timing applies to this case."""
    return enabled and case.flow == "allo" and case.dtype == "bf16"


def preview_case(case: GemmCase, args: argparse.Namespace) -> None:
    """Print the principal command that a dry run would execute."""
    case_work = work_path(case)
    print(
        f"{case.case_id}: {case.flow} {case.dtype} "
        f"M={case.M} N={case.N} K={case.K} "
        f"tile={case.m}x{case.n}x{case.k} columns={case.npu_columns}"
    )
    if case.flow == "allo":
        benchmark_on_failure = benchmark_on_validation_failure_for_case(
            case, args.benchmark_on_validation_failure
        )
        command = allo_worker_command(
            case,
            args.warmup,
            args.iterations,
            case_work / "allo.prj",
            benchmark_on_failure,
        )
    else:
        command = mlir_make_command(case, args.mlir_aie_root)
    print(f"  {render_command(command)}")


def run_experiments(args: argparse.Namespace) -> int:
    """Execute selected cases and regenerate result tables."""
    cases = generate_cases(args.flow, args.dtypes, args.Ms, args.Ns, args.Ks)
    if args.dry_run:
        for case in cases:
            preview_case(case, args)
        print(f"Dry run: {len(cases)} configuration(s)")
        return 0

    flows = expand_flows(args.flow)
    check_environment(flows, args.mlir_aie_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["NPU2"] = "0"
    failures = 0
    interrupted = False
    for index, case in enumerate(cases, start=1):
        result_file = record_path(output_dir, case)
        benchmark_on_failure = benchmark_on_validation_failure_for_case(
            case, args.benchmark_on_validation_failure
        )
        signature = case_signature(
            case, args.warmup, args.iterations, benchmark_on_failure
        )
        if not args.rerun and is_resumable(result_file, signature):
            print(f"[{index}/{len(cases)}] SKIP {case.case_id} (completed)")
            continue

        print(f"[{index}/{len(cases)}] RUN  {case.case_id}")
        case_work = work_path(case)
        case_work.mkdir(parents=True, exist_ok=True)
        case_log = log_path(output_dir, case)
        case_log.parent.mkdir(parents=True, exist_ok=True)
        case_log.write_text("", encoding="utf-8")
        record = new_record(
            case, args.warmup, args.iterations, output_dir, benchmark_on_failure
        )
        if case.flow == "allo":
            planned_command = allo_worker_command(
                case,
                args.warmup,
                args.iterations,
                case_work / "allo.prj",
                benchmark_on_failure,
            )
        else:
            planned_command = mlir_make_command(case, args.mlir_aie_root)
        record["commands"] = [render_command(planned_command)]
        atomic_write_json(result_file, record)
        started = time.monotonic()
        try:
            if case.flow == "allo":
                timings, commands, validation = run_allo_case(
                    case,
                    args.warmup,
                    args.iterations,
                    case_work,
                    case_log,
                    environment,
                    benchmark_on_failure,
                )
            else:
                timings, commands = run_mlir_aie_case(
                    case,
                    args.warmup,
                    args.iterations,
                    case_work,
                    case_log,
                    environment,
                    args.mlir_aie_root,
                )
                validation = "passed"
            if validation == "failed":
                record.update(
                    {
                        "status": "failed",
                        "validation": "failed",
                        "timed_validation_failure": True,
                        "timings_us": timings,
                        "commands": commands,
                        "error": (
                            "Output validation failed; timings were recorded because "
                            "--benchmark-on-validation-failure was enabled"
                        ),
                    }
                )
                print(f"TIMED VALIDATION FAILURE {case.case_id}", file=sys.stderr)
            else:
                record.update(
                    {
                        "status": "success",
                        "validation": "passed",
                        "timings_us": timings,
                        "commands": commands,
                    }
                )
        except KeyboardInterrupt:
            interrupted = True
            record.update(
                {
                    "status": "failed",
                    "validation": "interrupted",
                    "error": "Interrupted by user",
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures += 1
            details = traceback.format_exc()
            with case_log.open("a", encoding="utf-8") as log:
                log.write(f"\n{details}")
            record.update(
                {
                    "status": "failed",
                    "validation": "failed",
                    "error": str(exc),
                }
            )
            print(f"FAILED {case.case_id}: {exc}", file=sys.stderr)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["finished_at"] = utc_now()
            atomic_write_json(result_file, record)

        completed = record["status"] == "success" or is_timed_validation_failure(record)
        if completed and not args.keep_builds:
            safe_remove_work(case_work)
        if interrupted:
            failures += 1
            break
        if (
            record["status"] == "failed"
            and not is_timed_validation_failure(record)
            and args.fail_fast
        ):
            break

    paths = process_results(output_dir)
    print("Generated result tables:")
    for path in paths:
        print(f"  {path}")
    if not args.keep_builds and failures == 0:
        shared_host_root = DEFAULT_WORK_DIR / "mlir-aie-host"
        if shared_host_root.exists():
            safe_remove_work(shared_host_root)
    return 1 if failures else 0


def print_case_list(args: argparse.Namespace) -> int:
    """Print resolved configurations without touching the NPU."""
    cases = generate_cases(args.flow, args.dtypes, args.Ms, args.Ns, args.Ks)
    for case in cases:
        print(
            f"{case.case_id}: flow={case.flow} dtype={case.dtype} "
            f"M={case.M} N={case.N} K={case.K} "
            f"tile={case.m}x{case.n}x{case.k} columns={case.npu_columns}"
        )
    print(f"Total configurations: {len(cases)}")
    return 0


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add flow and Cartesian-product selection flags."""
    parser.add_argument("--flow", choices=(*FLOW_ORDER, "both"), required=True)
    parser.add_argument(
        "--dtype",
        dest="dtypes",
        choices=DEFAULT_DTYPES,
        nargs="+",
        default=list(DEFAULT_DTYPES),
        help="one or more input/output datatypes",
    )
    parser.add_argument(
        "--M",
        dest="Ms",
        type=int,
        choices=DEFAULT_SIZES,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="one or more M dimensions",
    )
    parser.add_argument(
        "--N",
        dest="Ns",
        type=int,
        choices=DEFAULT_SIZES,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="one or more N dimensions",
    )
    parser.add_argument(
        "--K",
        dest="Ks",
        type=int,
        choices=DEFAULT_SIZES,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="one or more K dimensions",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run and process NPU GEMM experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="build, validate, and benchmark selected configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_selection_arguments(run_parser)
    run_parser.add_argument("--warmup", type=int, default=20)
    run_parser.add_argument("--iterations", type=int, default=200)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run_parser.add_argument(
        "--benchmark-on-validation-failure",
        action="store_true",
        help=("time Allo bf16 after a comparison failure and mark it failed"),
    )
    run_parser.add_argument("--mlir-aie-root", type=Path, default=DEFAULT_MLIR_AIE_ROOT)
    run_parser.add_argument(
        "--rerun", action="store_true", help="rerun matching completed cases"
    )
    run_parser.add_argument(
        "--keep-builds",
        action="store_true",
        help="retain completed per-case build artifacts",
    )
    run_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first failed configuration",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print configurations and principal commands without executing",
    )
    run_parser.set_defaults(handler=run_experiments)

    list_parser = subparsers.add_parser(
        "list",
        help="list selected configurations without accessing hardware",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_selection_arguments(list_parser)
    list_parser.set_defaults(handler=print_case_list)

    process_parser = subparsers.add_parser(
        "process",
        help="regenerate CSV files from per-case JSON records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    process_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    def process_handler(args: argparse.Namespace) -> int:
        paths = process_results(args.output_dir)
        for path in paths:
            print(path)
        return 0

    process_parser.set_defaults(handler=process_handler)
    return parser


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate numeric arguments not expressible through argparse choices."""
    if args.command != "run":
        return
    if args.warmup < 0:
        raise ExperimentError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ExperimentError("--iterations must be positive")

    if args.benchmark_on_validation_failure and (
        "allo" not in expand_flows(args.flow) or "bf16" not in args.dtypes
    ):
        raise ExperimentError(
            "--benchmark-on-validation-failure requires an Allo bf16 selection"
        )


def build_allo_worker_parser() -> argparse.ArgumentParser:
    """Build the internal parser used to isolate Allo subprocess output."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dtype", choices=DEFAULT_DTYPES, required=True)
    for name in ("M", "N", "K", "m", "n", "k"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--columns", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--device-columns", type=int, choices=(1, 2, 4))
    parser.add_argument("--rows", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--benchmark-on-validation-failure", action="store_true")
    return parser


def allo_worker_main(argv: Sequence[str]) -> int:
    """Build, validate, and benchmark one Allo GEMM configuration."""
    args = build_allo_worker_parser().parse_args(argv)
    import allo.dataflow as df

    if args.benchmark_on_validation_failure and args.dtype != "bf16":
        raise ExperimentError(
            "--benchmark-on-validation-failure is supported only for Allo bf16"
        )
    # Imports stay local so list, dry-run, and result processing do not require
    # a built Allo/MLIR installation.
    # pylint: disable=import-outside-toplevel,import-error
    from allo.ir.types import bfloat16, int8, int16
    from allo.library.aie.modules.gemm import GEMM
    from ml_dtypes import bfloat16 as np_bfloat16

    type_map = {"int16": int16, "int8": int8, "bf16": bfloat16}
    Ty = type_map[args.dtype]
    top, mapping_primitives = GEMM(
        args.M,
        args.N,
        args.K,
        args.M // args.m,
        args.N // args.n,
        args.K // args.k,
        Ty,
        Ty,
        col_num=args.columns,
        row_num=args.rows,
    )
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    module = df.build(
        top,
        project=str(args.project),
        target="aie",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=args.warmup,
        num_iters=args.iterations,
        device_type=f"npu1_{args.device_columns or args.columns}col",
    )

    host_source = args.project / "test.cpp"
    instrumented = instrument_timing_source(host_source.read_text(encoding="utf-8"))
    host_source.write_text(instrumented, encoding="utf-8")
    subprocess.run(
        ["cmake", "--build", str(args.project / "build"), "--config", "Release"],
        check=True,
    )

    seed = (
        args.M * 73856093
        ^ args.N * 19349663
        ^ args.K * 83492791
        ^ DEFAULT_DTYPES.index(args.dtype)
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    if args.dtype == "int8":
        A = rng.integers(-4, 4, size=(args.M, args.K), dtype=np.int8)
        B = rng.integers(-4, 4, size=(args.K, args.N), dtype=np.int8)
        C = np.zeros((args.M, args.N), dtype=np.int8)
    elif args.dtype == "int16":
        A = rng.integers(-8, 8, size=(args.M, args.K), dtype=np.int16)
        B = rng.integers(-8, 8, size=(args.K, args.N), dtype=np.int16)
        C = np.zeros((args.M, args.N), dtype=np.int16)
    else:
        A = (rng.random((args.M, args.K), dtype=np.float32) * 0.1).astype(np_bfloat16)
        B = (rng.random((args.K, args.N), dtype=np.float32) * 0.1).astype(np_bfloat16)
        C = np.zeros((args.M, args.N), dtype=np_bfloat16)

    module.profile = False
    module(A, B, C)
    expected = A @ B
    try:
        if args.dtype == "bf16":
            np.testing.assert_allclose(
                C.astype(np.float32), expected.astype(np.float32), atol=1e-1
            )
        else:
            np.testing.assert_array_equal(C, expected)
    except AssertionError:
        if not args.benchmark_on_validation_failure:
            raise
        traceback.print_exc()
        print(ALLO_VALIDATION_FAILED, flush=True)
    else:
        print(ALLO_VALIDATION_PASSED, flush=True)

    C.fill(0)
    module.profile = True
    module(A, B, C)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entrypoint."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_allo-worker":
        return allo_worker_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        validate_cli_arguments(args)
        return args.handler(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
