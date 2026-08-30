#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and process bf16 GEMM experiments on partial NPU configurations."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import sys
import time
import traceback

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import gemm as base  # pylint: disable=wrong-import-position


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "gemm-partial-npu"
DEFAULT_WORK_DIR = base.DEFAULT_WORK_DIR / "partial-npu"
DEFAULT_COLUMNS = (1, 2, 4)
ALL_COLUMNS = (1, 2, 4, 8)
XDNA2_MATRIX_NS = (512, 1024, 2048)
VARIANT_ORDER = ("manual", "compiled", "compiled-full-io")
VARIANT_LABELS = {
    "manual": "Manual Template",
    "compiled": "Compiled",
    "compiled-full-io": "Compiled (Full I/O)",
}
EXPERIMENT_NAME = "gemm-partial-npu"
DTYPE = "bf16"
TILE_M = 64
TILE_N = 64
TILE_K = 64


def default_output_dir(device: str = base.DEFAULT_DEVICE) -> Path:
    """Return the isolated partial-GEMM result directory for a device."""
    if device == "xdna2":
        return EXPERIMENT_DIR / "results" / "gemm-partial-npu-xdna2"
    base.device_config(device)
    return DEFAULT_OUTPUT_DIR


def default_columns(device: str = base.DEFAULT_DEVICE) -> tuple[int, ...]:
    """Return default logical compute widths for a device."""
    config = base.device_config(device)
    return tuple(reversed(config.column_choices))


def default_matrix_ns(device: str = base.DEFAULT_DEVICE) -> tuple[int, ...]:
    """Return default N dimensions whose tiles divide each selected width."""
    base.device_config(device)
    return XDNA2_MATRIX_NS if device == "xdna2" else base.DEFAULT_SIZES


@dataclass(frozen=True)
class PartialNpuCase:
    """One physical partial-NPU GEMM experiment configuration."""

    variant: str
    compute_columns: int
    device_columns: int
    M: int
    N: int
    K: int
    device: str = base.DEFAULT_DEVICE

    @property
    def flow(self) -> str:
        """Return the backend used by this physical configuration."""
        return "mlir-aie" if self.variant == "manual" else "allo"

    @property
    def plot_series(self) -> tuple[str, ...]:
        """Return the logical plot series supplied by this execution."""
        if (
            self.variant == "compiled"
            and self.compute_columns == base.device_config(self.device).max_columns
        ):
            return "compiled", "compiled-full-io"
        return (self.variant,)

    @property
    def mapping_columns(self) -> int:
        """Return the Allo mapping column count."""
        if self.variant == "compiled" and self.compute_columns == 1:
            return 2
        return self.compute_columns

    @property
    def mapping_rows(self) -> int:
        """Return the Allo mapping row count."""
        if self.variant == "compiled" and self.compute_columns == 1:
            return 2
        return 4

    @property
    def case_id(self) -> str:
        """Return a stable result and work identifier."""
        return (
            f"{self.variant}_{DTYPE}_M{self.M}_N{self.N}_K{self.K}_"
            f"{self.compute_columns}compute_{self.device_columns}device"
        )

    def as_gemm_case(self) -> base.GemmCase:
        """Return the shared runner representation for this case."""
        return base.GemmCase(
            self.flow,
            DTYPE,
            self.M,
            self.N,
            self.K,
            TILE_M,
            TILE_N,
            TILE_K,
            self.mapping_columns,
        )


def unique(values: Iterable[object]) -> list[object]:
    """Remove duplicates while preserving command-line order."""
    return list(dict.fromkeys(values))


def physical_variants(
    columns: int,
    variants: Sequence[str],
    device: str = base.DEFAULT_DEVICE,
) -> tuple[str, ...]:
    """Resolve requested plot variants to deduplicated physical executions."""
    full_width = base.device_config(device).max_columns
    selected = set(variants)
    resolved = []
    if "manual" in selected:
        resolved.append("manual")
    if "compiled" in selected or (
        columns == full_width and "compiled-full-io" in selected
    ):
        resolved.append("compiled")
    if columns < full_width and "compiled-full-io" in selected:
        resolved.append("compiled-full-io")
    return tuple(resolved)


def device_columns_for(
    variant: str,
    compute_columns: int,
    device: str = base.DEFAULT_DEVICE,
) -> int:
    """Return the physical device width for one variant."""
    if variant == "compiled-full-io":
        return base.device_config(device).max_columns
    return compute_columns


def generate_cases(
    variants: Sequence[str],
    columns: Sequence[int],
    matrix_ms: Sequence[int],
    matrix_ns: Sequence[int],
    matrix_ks: Sequence[int],
    device: str = base.DEFAULT_DEVICE,
) -> list[PartialNpuCase]:
    """Generate requested physical configurations in deterministic order."""
    config = base.device_config(device)
    cases = []
    selected_variants = [str(value) for value in unique(variants)]
    for compute_columns in unique(columns):
        compute_columns = int(compute_columns)
        if compute_columns not in config.column_choices:
            raise ValueError(
                f"{device.upper()} does not support --columns {compute_columns}"
            )
        invalid_ns = [
            int(N)
            for N in unique(matrix_ns)
            if int(N) % (TILE_N * compute_columns) != 0
        ]
        if invalid_ns:
            values = ", ".join(str(value) for value in invalid_ns)
            raise ValueError(
                f"N={values} cannot be divided across {compute_columns} "
                f"compute columns with tile n={TILE_N}"
            )
        for variant in physical_variants(compute_columns, selected_variants, device):
            for M, N, K in itertools.product(
                unique(matrix_ms), unique(matrix_ns), unique(matrix_ks)
            ):
                cases.append(
                    PartialNpuCase(
                        variant,
                        compute_columns,
                        device_columns_for(variant, compute_columns, device),
                        int(M),
                        int(N),
                        int(K),
                        device,
                    )
                )
    return cases


def logical_case_count(cases: Sequence[PartialNpuCase]) -> int:
    """Count logical plotted points after expanding physical aliases."""
    return sum(len(case.plot_series) for case in cases)


def case_signature(case: PartialNpuCase, warmup: int, iterations: int) -> dict:
    """Return fields that determine whether a partial-NPU result is resumable."""
    config = base.device_config(case.device)
    signature = {
        "schema_version": base.SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "variant": case.variant,
        "flow": case.flow,
        "dtype": DTYPE,
        "M": case.M,
        "N": case.N,
        "K": case.K,
        "m": TILE_M,
        "n": TILE_N,
        "k": TILE_K,
        "compute_columns": case.compute_columns,
        "mapping_columns": case.mapping_columns,
        "mapping_rows": case.mapping_rows,
        "device_columns": case.device_columns,
        "device": case.device,
        "target": config.target,
        "backend_target": config.backend_target(case.flow, case.device_columns),
        "npu2": config.npu2,
        "warmup": warmup,
        "iterations": iterations,
        "benchmark_on_validation_failure": case.flow == "allo",
    }
    return signature


def record_path(output_dir: Path, case: PartialNpuCase) -> Path:
    """Return the per-case JSON path."""
    return (
        output_dir
        / "cases"
        / case.variant
        / f"{case.compute_columns}col"
        / f"{case.case_id}.json"
    )


def log_path(output_dir: Path, case: PartialNpuCase) -> Path:
    """Return the per-case build and execution log path."""
    return (
        output_dir
        / "logs"
        / case.variant
        / f"{case.compute_columns}col"
        / f"{case.case_id}.log"
    )


def work_path(case: PartialNpuCase) -> Path:
    """Return the device-isolated work directory for a physical case."""
    return DEFAULT_WORK_DIR / case.device / case.variant / case.case_id


def device_override(case: PartialNpuCase) -> int | None:
    """Return a device-width override only when it differs from mapping width."""
    if case.device_columns == case.mapping_columns:
        return None
    return case.device_columns


def allo_command(
    case: PartialNpuCase, warmup: int, iterations: int, project: Path
) -> list[str]:
    """Build the shared Allo worker command for one partial-NPU case."""
    return base.allo_worker_command(
        case.as_gemm_case(),
        warmup,
        iterations,
        project,
        benchmark_on_validation_failure=True,
        device_columns=device_override(case),
        mapping_rows=case.mapping_rows if case.mapping_rows != 4 else None,
        device=case.device,
    )


def planned_command(
    case: PartialNpuCase,
    warmup: int,
    iterations: int,
    case_work: Path,
    mlir_aie_root: Path,
) -> list[str]:
    """Return the principal command for logs and dry-run output."""
    if case.flow == "allo":
        return allo_command(case, warmup, iterations, case_work / "allo.prj")
    return base.mlir_make_command(case.as_gemm_case(), mlir_aie_root, case.device)


def new_record(
    case: PartialNpuCase, warmup: int, iterations: int, output_dir: Path
) -> dict:
    """Create a running per-case result record."""
    config = base.device_config(case.device)
    case_log = log_path(output_dir, case)
    try:
        stored_log_path = str(case_log.relative_to(output_dir))
    except ValueError:
        stored_log_path = str(case_log)
    return {
        "schema_version": base.SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "case_id": case.case_id,
        "variant": case.variant,
        "plot_series": list(case.plot_series),
        "flow": case.flow,
        "dtype": DTYPE,
        "M": case.M,
        "N": case.N,
        "K": case.K,
        "m": TILE_M,
        "n": TILE_N,
        "k": TILE_K,
        "compute_columns": case.compute_columns,
        "device_columns": case.device_columns,
        "npu_columns": case.mapping_columns,
        "mapping_columns": case.mapping_columns,
        "mapping_rows": case.mapping_rows,
        "device": case.device,
        "target": config.target,
        "backend_target": config.backend_target(case.flow, case.device_columns),
        "npu2": config.npu2,
        "output_dtype": DTYPE,
        "warmup": warmup,
        "iterations": iterations,
        "signature": case_signature(case, warmup, iterations),
        "benchmark_on_validation_failure": case.flow == "allo",
        "status": "running",
        "validation": "not_run",
        "timed_validation_failure": False,
        "timings_us": [],
        "commands": [],
        "started_at": base.utc_now(),
        "finished_at": None,
        "elapsed_seconds": None,
        "error": "",
        "log_path": stored_log_path,
    }


def load_records(output_dir: Path) -> list[dict]:
    """Load and deterministically order partial-NPU JSON records."""
    records = []
    for path in sorted((output_dir / "cases").glob("**/*.json")):
        with path.open(encoding="utf-8") as source:
            record = json.load(source)
        if record.get("schema_version") != base.SCHEMA_VERSION:
            raise base.ExperimentError(f"Unsupported result schema in {path}")
        if record.get("experiment") != EXPERIMENT_NAME:
            raise base.ExperimentError(f"Unexpected experiment record in {path}")
        record.setdefault("mapping_columns", record["compute_columns"])
        record.setdefault("mapping_rows", 4)
        records.append(record)
    return sorted(
        records,
        key=lambda item: (
            ALL_COLUMNS.index(item["compute_columns"]),
            VARIANT_ORDER.index(item["variant"]),
            item["M"],
            item["N"],
            item["K"],
        ),
    )


def csv_value(record: dict, field: str):
    """Return a scalar CSV representation of one record field."""
    value = record.get(field, "")
    if field == "plot_series" and isinstance(value, list):
        return ";".join(value)
    return value


def process_results(output_dir: Path) -> tuple[Path, Path, Path]:
    """Regenerate partial-NPU raw, filtered, and summary CSV files."""
    output_dir = output_dir.resolve()
    records = load_records(output_dir)
    timing_fields = [
        "case_id",
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
        "m",
        "n",
        "k",
        "compute_columns",
        "device_columns",
        "sample_index",
        "time_us",
        "mapping_columns",
        "mapping_rows",
        "gflops",
        "is_outlier",
    ]
    summary_fields = [
        "case_id",
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
        "m",
        "n",
        "k",
        "compute_columns",
        "device_columns",
        "warmup",
        "iterations",
        "mapping_columns",
        "mapping_rows",
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
        filter_result = base.tukey_filter(timings)
        mask = filter_result["mask"]
        completed = record.get(
            "status"
        ) == "success" or base.is_timed_validation_failure(record)
        for index, value in enumerate(timings):
            row = {
                field: csv_value(record, field)
                for field in timing_fields
                if field not in {"sample_index", "time_us", "gflops", "is_outlier"}
            }
            row.update(
                {
                    "sample_index": index,
                    "time_us": value,
                    "gflops": base.gflops_for(record, value),
                    "is_outlier": not mask[index],
                }
            )
            raw_rows.append(row)
            if completed and mask[index]:
                filtered_rows.append(row.copy())

        filtered = filter_result["filtered"] if completed else []
        summary = {
            field: csv_value(record, field)
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
        summary.update(base.describe_timings(record, timings, "raw"))
        summary.update(base.describe_timings(record, filtered, "filtered"))
        summary_rows.append(summary)

    raw_path = output_dir / "raw_timings.csv"
    filtered_path = output_dir / "filtered_timings.csv"
    summary_path = output_dir / "summary.csv"
    base.atomic_write_csv(raw_path, timing_fields, raw_rows)
    base.atomic_write_csv(filtered_path, timing_fields, filtered_rows)
    base.atomic_write_csv(summary_path, summary_fields, summary_rows)
    return raw_path, filtered_path, summary_path


def preview_case(case: PartialNpuCase, args: argparse.Namespace) -> None:
    """Print one resolved physical configuration and its principal command."""
    supplied = ",".join(VARIANT_LABELS[item] for item in case.plot_series)
    print(
        f"{case.case_id}: {supplied}; compute={case.compute_columns}x4 "
        f"mapping={case.mapping_rows}x{case.mapping_columns} "
        f"device={case.device_columns}col M={case.M} N={case.N} K={case.K}"
    )
    command = planned_command(
        case,
        args.warmup,
        args.iterations,
        work_path(case),
        args.mlir_aie_root,
    )
    print(f"  {base.render_command(command)}")


def run_physical_case(
    case: PartialNpuCase,
    args: argparse.Namespace,
    case_work: Path,
    case_log: Path,
    environment: dict[str, str],
) -> tuple[list[float], list[str], str]:
    """Execute one physical configuration through the shared runner."""
    gemm_case = case.as_gemm_case()
    if case.flow == "allo":
        return base.run_allo_case(
            gemm_case,
            args.warmup,
            args.iterations,
            case_work,
            case_log,
            environment,
            benchmark_on_validation_failure=True,
            device_columns=device_override(case),
            mapping_rows=case.mapping_rows if case.mapping_rows != 4 else None,
            device=case.device,
        )
    timings, commands = base.run_mlir_aie_case(
        gemm_case,
        args.warmup,
        args.iterations,
        case_work,
        case_log,
        environment,
        args.mlir_aie_root,
        case.device,
    )
    return timings, commands, "passed"


def run_experiments(args: argparse.Namespace) -> int:
    """Execute selected physical cases and regenerate aggregate tables."""
    config = base.device_config(args.device)
    columns = args.columns or default_columns(args.device)
    matrix_ns = args.Ns or default_matrix_ns(args.device)
    cases = generate_cases(
        args.variants,
        columns,
        args.Ms,
        matrix_ns,
        args.Ks,
        args.device,
    )
    if args.dry_run:
        for case in cases:
            preview_case(case, args)
        print(
            f"Dry run: {len(cases)} physical configuration(s), "
            f"{logical_case_count(cases)} plotted series point(s)"
        )
        return 0

    flows = tuple(
        flow for flow in base.FLOW_ORDER if any(c.flow == flow for c in cases)
    )
    base.check_environment(flows, args.mlir_aie_root)
    output_dir = (args.output_dir or default_output_dir(args.device)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["NPU2"] = config.npu2
    failures = 0
    interrupted = False
    try:
        for index, case in enumerate(cases, start=1):
            result_file = record_path(output_dir, case)
            signature = case_signature(case, args.warmup, args.iterations)
            case_work = work_path(case)
            if not args.rerun and base.is_resumable(result_file, signature):
                if not args.keep_builds:
                    base.safe_remove_work(case_work)
                print(f"[{index}/{len(cases)}] SKIP {case.case_id} (completed)")
                continue

            print(f"[{index}/{len(cases)}] RUN  {case.case_id}")
            base.safe_remove_work(case_work)
            case_work.mkdir(parents=True, exist_ok=True)
            case_log = log_path(output_dir, case)
            case_log.parent.mkdir(parents=True, exist_ok=True)
            case_log.write_text("", encoding="utf-8")
            record = new_record(case, args.warmup, args.iterations, output_dir)
            command = planned_command(
                case, args.warmup, args.iterations, case_work, args.mlir_aie_root
            )
            record["commands"] = [base.render_command(command)]
            base.atomic_write_json(result_file, record)
            started = time.monotonic()
            try:
                timings, commands, validation = run_physical_case(
                    case, args, case_work, case_log, environment
                )
                if validation == "failed":
                    record.update(
                        {
                            "status": "failed",
                            "validation": "failed",
                            "timed_validation_failure": True,
                            "timings_us": timings,
                            "commands": commands,
                            "error": (
                                "Output validation failed; timings were recorded "
                                "automatically for the partial-NPU bf16 experiment"
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
                record["finished_at"] = base.utc_now()
                base.atomic_write_json(result_file, record)
                if not args.keep_builds:
                    base.safe_remove_work(case_work)

            if interrupted:
                failures += 1
                break
            completed = record[
                "status"
            ] == "success" or base.is_timed_validation_failure(record)
            if record["status"] == "failed" and not completed and args.fail_fast:
                break

        paths = process_results(output_dir)
        print("Generated result tables:")
        for path in paths:
            print(f"  {path}")
        return 1 if failures else 0
    finally:
        shared_host_root = base.shared_host_path(args.device)
        if shared_host_root.exists():
            base.safe_remove_work(shared_host_root)


def print_case_list(args: argparse.Namespace) -> int:
    """Print resolved physical cases without accessing hardware."""
    columns = args.columns or default_columns(args.device)
    matrix_ns = args.Ns or default_matrix_ns(args.device)
    cases = generate_cases(
        args.variants, columns, args.Ms, matrix_ns, args.Ks, args.device
    )
    for case in cases:
        supplied = ",".join(case.plot_series)
        print(
            f"{case.case_id}: flow={case.flow} series={supplied} "
            f"compute={case.compute_columns}x4 "
            f"mapping={case.mapping_rows}x{case.mapping_columns} "
            f"device={case.device_columns}col M={case.M} N={case.N} K={case.K}"
        )
    print(f"Total physical configurations: {len(cases)}")
    print(f"Total plotted series points: {logical_case_count(cases)}")
    return 0


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add variant, column, and Cartesian-product dimension selectors."""
    parser.add_argument(
        "--device", choices=base.DEVICE_CHOICES, default=base.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--variant",
        dest="variants",
        choices=VARIANT_ORDER,
        nargs="+",
        default=list(VARIANT_ORDER),
    )
    parser.add_argument(
        "--columns",
        type=int,
        choices=ALL_COLUMNS,
        nargs="+",
        default=None,
        help="logical compute-tile columns (device-specific when omitted)",
    )
    for name in ("M", "K"):
        parser.add_argument(
            f"--{name}",
            dest=f"{name}s",
            type=int,
            choices=base.DEFAULT_SIZES,
            nargs="+",
            default=list(base.DEFAULT_SIZES),
        )
    parser.add_argument(
        "--N",
        dest="Ns",
        type=int,
        choices=base.DEFAULT_SIZES,
        nargs="+",
        default=None,
        help="N dimensions (device-specific when omitted)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the public command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark bf16 GEMM on partial NPU compute configurations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list physical run configurations")
    add_selection_arguments(list_parser)
    list_parser.set_defaults(handler=print_case_list)

    run_parser = subparsers.add_parser("run", help="run selected configurations")
    add_selection_arguments(run_parser)
    run_parser.add_argument("--warmup", type=int, default=20)
    run_parser.add_argument("--iterations", type=int, default=200)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (device-specific when omitted)",
    )
    run_parser.add_argument(
        "--mlir-aie-root", type=Path, default=base.DEFAULT_MLIR_AIE_ROOT
    )
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--keep-builds", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=run_experiments)

    process_parser = subparsers.add_parser(
        "process", help="regenerate aggregate CSV files from case records"
    )
    process_parser.add_argument(
        "--device", choices=base.DEVICE_CHOICES, default=base.DEFAULT_DEVICE
    )
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (device-specific when omitted)",
    )

    def process_handler(args: argparse.Namespace) -> int:
        output_dir = args.output_dir or default_output_dir(args.device)
        paths = process_results(output_dir)
        for path in paths:
            print(path)
        return 0

    process_parser.set_defaults(handler=process_handler)
    return parser


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Reject nonsensical timing settings before any case is started."""
    if args.command != "run":
        return
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_cli_arguments(args)
        return args.handler(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
