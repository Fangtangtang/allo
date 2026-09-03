#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run and process end-to-end attention experiments on Ryzen AI NPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from allo.ir.types import Stream, bfloat16
from allo.memory import Layout

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import gemm as base  # pylint: disable=wrong-import-position,wrong-import-order

EXPERIMENT_NAME = "attention"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / EXPERIMENT_NAME
DEFAULT_WORK_DIR = EXPERIMENT_DIR / ".work" / EXPERIMENT_NAME
DEFAULT_SEQ_LENS = (64, 128, 256, 512, 1024, 2048)
IMPLEMENTATION_ORDER = ("baseline", "flash")
IMPLEMENTATION_LABELS = {
    "baseline": "Unfused Baseline",
    "flash": "Fused FlashAttention",
}
HEAD_DIM = 64
Q_CHUNK_SIZE = 32
KV_CHUNK_SIZE = 32
DTYPE = "bf16"
TIMING_SCOPE = "end-to-end"
ATTENTION_TIMING_VERSION = 2
NPU_TIMING_SCOPE = "test.cpp kernel launch through run.wait()"

BF16_TYPE = bfloat16
GEMM_LAYOUT_A = [Layout.Shard(1), Layout.Shard(0)]
GEMM_LAYOUT_B = [Layout.Shard(0), Layout.Shard(2)]
GEMM_LAYOUT_C = [Layout.Shard(1), Layout.Shard(2)]
SCALE_LAYOUT = [Layout.Shard(0), Layout.Shard(1)]
SOFTMAX_LAYOUT = [Layout.Shard(0), Layout.Replicate]
TIMING_PREFIX = "ATTENTION_E2E_SAMPLE_US="
NPU_TIMING_PREFIX = "NPU execution time: "
VALIDATION_PASSED = "ATTENTION_VALIDATION=PASSED"
VALIDATION_FAILED = "ATTENTION_VALIDATION=FAILED"
INFEASIBLE_BASELINE_SEQ_LEN = 2048
INFEASIBLE_BASELINE_REASON = (
    "The unfused baseline at sequence length 2048 is intentionally not executed"
)


class ExperimentError(RuntimeError):
    """An expected attention experiment failure."""


def default_output_dir(device: str = base.DEFAULT_DEVICE) -> Path:
    """Return the device-specific attention result directory."""
    if device == "xdna2":
        return EXPERIMENT_DIR / "results" / "attention-xdna2"
    base.device_config(device)
    return DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class AttentionCase:
    """One attention implementation and sequence-length configuration."""

    implementation: str
    seq_len: int
    device: str = base.DEFAULT_DEVICE

    @property
    def mapping_rows(self) -> int:
        """Return the physical compute-row count."""
        return 4

    @property
    def mapping_columns(self) -> int:
        """Return the physical compute-column count."""
        return base.device_config(self.device).max_columns

    @property
    def compute_slots(self) -> int:
        """Return the number of compute tiles in the selected device."""
        return self.mapping_rows * self.mapping_columns

    @property
    def kernel_count(self) -> int:
        """Return the number of separately invoked AIE modules."""
        return 4 if self.implementation == "baseline" else 1

    @property
    def npu_aggregation(self) -> str:
        """Return how per-kernel test.cpp timings form one attention sample."""
        return (
            "sum-of-4-kernels" if self.implementation == "baseline" else "single-kernel"
        )

    @property
    def infeasible(self) -> bool:
        """Return whether the baseline case is intentionally not executed."""
        return (
            self.implementation == "baseline"
            and self.seq_len == INFEASIBLE_BASELINE_SEQ_LEN
        )

    @property
    def case_id(self) -> str:
        """Return a stable result and work identifier."""
        return f"{self.implementation}_{DTYPE}_N{self.seq_len}_D{HEAD_DIM}"


def unique(values: Iterable[object]) -> list[object]:
    """Remove duplicates while preserving input order."""
    return list(dict.fromkeys(values))


def expand_implementations(implementation: str) -> tuple[str, ...]:
    """Expand the public implementation selector."""
    if implementation == "both":
        return IMPLEMENTATION_ORDER
    if implementation in IMPLEMENTATION_ORDER:
        return (implementation,)
    raise ValueError(f"Unsupported implementation: {implementation}")


def generate_cases(
    implementation: str,
    seq_lens: Sequence[int],
    device: str = base.DEFAULT_DEVICE,
) -> list[AttentionCase]:
    """Generate selected configurations in deterministic order."""
    base.device_config(device)
    selected_lengths = [int(value) for value in unique(seq_lens)]
    invalid = [value for value in selected_lengths if value not in DEFAULT_SEQ_LENS]
    if invalid:
        values = ", ".join(str(value) for value in invalid)
        raise ValueError(f"Unsupported sequence length(s): {values}")
    return [
        AttentionCase(selected, seq_len, device)
        for selected in expand_implementations(implementation)
        for seq_len in selected_lengths
    ]


def case_signature(case: AttentionCase, warmup: int, iterations: int) -> dict:
    """Return the fields determining whether a result is resumable."""
    config = base.device_config(case.device)
    return {
        "schema_version": base.SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        **asdict(case),
        "dtype": DTYPE,
        "head_dim": HEAD_DIM,
        "q_chunk_size": Q_CHUNK_SIZE,
        "kv_chunk_size": KV_CHUNK_SIZE,
        "mapping_rows": case.mapping_rows,
        "mapping_columns": case.mapping_columns,
        "compute_slots": case.compute_slots,
        "kernel_count": case.kernel_count,
        "timing_scope": TIMING_SCOPE,
        "attention_timing_version": ATTENTION_TIMING_VERSION,
        "npu_timing_scope": NPU_TIMING_SCOPE,
        "npu_aggregation": case.npu_aggregation,
        "device": case.device,
        "target": config.target,
        "backend_target": config.allo_device_type(case.mapping_columns),
        "npu2": config.npu2,
        "warmup": warmup,
        "iterations": iterations,
    }


def record_path(output_dir: Path, case: AttentionCase) -> Path:
    """Return the per-case JSON path."""
    return output_dir / "cases" / case.implementation / f"{case.case_id}.json"


def log_path(output_dir: Path, case: AttentionCase) -> Path:
    """Return the per-case log path."""
    return output_dir / "logs" / case.implementation / f"{case.case_id}.log"


def work_path(case: AttentionCase) -> Path:
    """Return the device-isolated build directory for a case."""
    return DEFAULT_WORK_DIR / case.device / case.implementation / case.case_id


def safe_remove_work(path: Path) -> None:
    """Remove only a validated descendant of the attention work root."""
    resolved_root = DEFAULT_WORK_DIR.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ExperimentError(f"Refusing to remove unsafe work path: {resolved_path}")
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def new_record(
    case: AttentionCase, warmup: int, iterations: int, output_dir: Path
) -> dict:
    """Create a running result record."""
    config = base.device_config(case.device)
    case_log = log_path(output_dir, case)
    try:
        stored_log = str(case_log.relative_to(output_dir))
    except ValueError:
        stored_log = str(case_log)
    return {
        "schema_version": base.SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "case_id": case.case_id,
        **asdict(case),
        "dtype": DTYPE,
        "head_dim": HEAD_DIM,
        "q_chunk_size": Q_CHUNK_SIZE,
        "kv_chunk_size": KV_CHUNK_SIZE,
        "mapping_rows": case.mapping_rows,
        "mapping_columns": case.mapping_columns,
        "compute_slots": case.compute_slots,
        "kernel_count": case.kernel_count,
        "timing_scope": TIMING_SCOPE,
        "attention_timing_version": ATTENTION_TIMING_VERSION,
        "npu_timing_scope": NPU_TIMING_SCOPE,
        "npu_aggregation": case.npu_aggregation,
        "target": config.target,
        "backend_target": config.allo_device_type(case.mapping_columns),
        "npu2": config.npu2,
        "warmup": warmup,
        "iterations": iterations,
        "signature": case_signature(case, warmup, iterations),
        "status": "running",
        "validation": "not_run",
        "timed_validation_failure": False,
        "timings_us": [],
        "npu_timings_us": [],
        "commands": [],
        "started_at": base.utc_now(),
        "finished_at": None,
        "elapsed_seconds": None,
        "error": "",
        "log_path": stored_log,
    }


def is_timed_validation_failure(record: dict) -> bool:
    """Return whether a baseline comparison failed after timings were recorded."""
    return (
        record.get("implementation") == "baseline"
        and record.get("status") == "failed"
        and record.get("validation") == "failed"
        and record.get("timed_validation_failure") is True
    )


def is_infeasible_record(record: dict) -> bool:
    """Return whether a record is the intentionally skipped baseline case."""
    return (
        record.get("implementation") == "baseline"
        and record.get("seq_len") == INFEASIBLE_BASELINE_SEQ_LEN
        and record.get("status") == "infeasible"
        and record.get("validation") == "not_run"
        and not record.get("timings_us")
        and not record.get("npu_timings_us")
    )


def is_resumable(record_file: Path, signature: dict) -> bool:
    """Return whether a completed record exactly matches this request."""
    if not record_file.is_file():
        return False
    try:
        with record_file.open(encoding="utf-8") as source:
            record = json.load(source)
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("signature") != signature:
        return False
    if is_infeasible_record(record):
        return True
    completed = (
        record.get("status") == "success" and record.get("validation") == "passed"
    ) or is_timed_validation_failure(record)
    timings = record.get("timings_us", [])
    npu_timings = record.get("npu_timings_us", [])
    if not completed or not len(timings) == len(npu_timings) == signature["iterations"]:
        return False
    try:
        derive_extra_timings(timings, npu_timings)
    except ExperimentError:
        return False
    return True


def worker_command(
    case: AttentionCase, warmup: int, iterations: int, project_root: Path
) -> list[str]:
    """Build the isolated worker command for one case."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--device",
        case.device,
        "--implementation",
        case.implementation,
        "--seq-len",
        str(case.seq_len),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--project-root",
        str(project_root),
    ]


def parse_sample_timings(output: str, expected_count: int) -> list[float]:
    """Parse positive machine-readable end-to-end samples."""
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
            raise ExperimentError(f"Invalid end-to-end timing: {value_text}")
        timings.append(value)
    if len(timings) != expected_count:
        raise ExperimentError(
            f"Expected {expected_count} end-to-end samples, captured {len(timings)}"
        )
    return timings


def parse_npu_timings(
    output: str,
    case: AttentionCase,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Parse and aggregate generated test.cpp kernel timings."""
    kernel_timings = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith(NPU_TIMING_PREFIX):
            continue
        payload = stripped[len(NPU_TIMING_PREFIX) :]
        if not payload.endswith("us"):
            raise ExperimentError(f"Malformed NPU timing line: {stripped}")
        value_text = payload[:-2]
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ExperimentError(f"Malformed NPU timing line: {stripped}") from exc
        if not math.isfinite(value) or value <= 0:
            raise ExperimentError(f"Invalid NPU timing: {value_text}")
        kernel_timings.append(value)

    expected_count = (1 + warmup + iterations) * case.kernel_count
    if len(kernel_timings) != expected_count:
        raise ExperimentError(
            f"Expected {expected_count} test.cpp NPU timings, "
            f"captured {len(kernel_timings)}"
        )
    recorded = kernel_timings[(1 + warmup) * case.kernel_count :]
    return [
        sum(recorded[offset : offset + case.kernel_count])
        for offset in range(0, len(recorded), case.kernel_count)
    ]


def derive_extra_timings(
    timings: Sequence[float], npu_timings: Sequence[float]
) -> list[float]:
    """Validate paired E2E/NPU samples and return their non-NPU portions."""
    if len(timings) != len(npu_timings):
        raise ExperimentError(
            f"E2E/NPU timing count mismatch: {len(timings)} != {len(npu_timings)}"
        )
    extra_timings = []
    for index, (timing, npu_timing) in enumerate(zip(timings, npu_timings)):
        try:
            e2e_value = float(timing)
            npu_value = float(npu_timing)
        except (TypeError, ValueError) as exc:
            raise ExperimentError(
                f"Malformed E2E/NPU timing pair at sample {index}"
            ) from exc
        if not math.isfinite(e2e_value) or e2e_value <= 0:
            raise ExperimentError(f"Invalid E2E timing at sample {index}: {timing}")
        if not math.isfinite(npu_value) or npu_value <= 0:
            raise ExperimentError(f"Invalid NPU timing at sample {index}: {npu_timing}")
        if npu_value > e2e_value:
            raise ExperimentError(
                f"NPU timing exceeds E2E timing at sample {index}: "
                f"{npu_value} > {e2e_value}"
            )
        extra_timings.append(e2e_value - npu_value)
    return extra_timings


def measure_complete_attention(
    run_once: Callable[[], None],
    warmup: int,
    iterations: int,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> list[float]:
    """Warm up and time complete attention invocations in microseconds."""
    for _ in range(warmup):
        run_once()
    samples = []
    for _ in range(iterations):
        started = clock_ns()
        run_once()
        elapsed_us = (clock_ns() - started) / 1000.0
        if not math.isfinite(elapsed_us) or elapsed_us <= 0:
            raise ExperimentError(f"Invalid measured end-to-end time: {elapsed_us}")
        samples.append(elapsed_us)
    return samples


def run_baseline_once(modules: Sequence[Callable[[], None]]) -> None:
    """Invoke the four unfused stages in order."""
    if len(modules) != 4:
        raise ValueError("The unfused baseline requires exactly four modules")
    for module in modules:
        module()


def gemm_mapping_primitives(
    pk_size: int,
    pm_size: int,
    pn_size: int,
    row_num: int,
    col_num: int,
    prefix: str = "gemm",
) -> list[tuple[str, list]]:
    """Map split-K GEMM tiles over a bounded physical grid."""
    primitives: list[tuple[str, list]] = []
    bases: list[list[str]] = []
    for pm in range(pm_size):
        bases.append([])
        for pn in range(pn_size):
            base_name = f"{prefix}_0_{pm}_{pn}"
            for pk in range(1, pk_size):
                next_name = f"{prefix}_{pk}_{pm}_{pn}"
                primitives.append(("chain", [base_name, next_name]))
                base_name += f"-{next_name}"
            bases[pm].append(base_name)

    active_rows = min(row_num, pm_size)
    active_cols = min(col_num, pn_size)
    if active_rows == 0 or active_cols == 0:
        return primitives
    if pm_size > active_rows or pn_size > active_cols:
        for row in range(active_rows):
            for col in range(active_cols):
                nodes = [
                    bases[pm][pn]
                    for pm in range(row, pm_size, active_rows)
                    for pn in range(col, pn_size, active_cols)
                ]
                if len(nodes) > 1:
                    primitives.append(("bundle", nodes))
    return primitives


def grid_mapping_primitives(
    prefix: str, p0_size: int, p1_size: int, row_num: int, col_num: int
) -> list[tuple[str, list]]:
    """Bundle a logical two-dimensional kernel grid over physical tiles."""
    active_rows = min(row_num, p0_size)
    active_cols = min(col_num, p1_size)
    if active_rows == 0 or active_cols == 0:
        return []
    primitives = []
    if p0_size > active_rows or p1_size > active_cols:
        for row in range(active_rows):
            for col in range(active_cols):
                nodes = [
                    f"{prefix}_{p0}_{p1}"
                    for p0 in range(row, p0_size, active_rows)
                    for p1 in range(col, p1_size, active_cols)
                ]
                if len(nodes) > 1:
                    primitives.append(("bundle", nodes))
    return primitives


def linear_mapping_primitives(
    prefix: str, logical_size: int, compute_slots: int
) -> list[tuple[str, list]]:
    """Bundle a logical one-dimensional grid over all compute tiles."""
    if logical_size <= compute_slots:
        return []
    primitives = []
    for slot in range(compute_slots):
        nodes = [
            f"{prefix}_{index}" for index in range(slot, logical_size, compute_slots)
        ]
        if len(nodes) > 1:
            primitives.append(("bundle", nodes))
    return primitives


def softmax_wrapper_source(seq_len: int, kernel_path: Path) -> tuple[str, str]:
    """Return a fixed-shape wrapper around the reusable softmax template."""
    if seq_len not in DEFAULT_SEQ_LENS:
        raise ValueError(f"Unsupported softmax sequence length: {seq_len}")
    top = f"attention_softmax_bf16_{seq_len}"
    source = f"""// Generated by aie-experiments/attention.py
#include "{kernel_path.resolve().as_posix()}"

extern "C" {{
void {top}(bfloat16 input[4][{seq_len}], bfloat16 output[4][{seq_len}]) {{
  for (int row = 0; row < 4; ++row) {{
    softmax_simple_bf16<{seq_len}>(&input[row][0], &output[row][0]);
  }}
}}
}}
"""
    return top, source


def write_softmax_wrapper(
    seq_len: int, device: str, destination: Path
) -> tuple[str, Path]:
    """Write the architecture-specific, fixed-shape softmax wrapper."""
    base.device_config(device)
    kernel_name = "softmax_bf16_aie2p.cc" if device == "xdna2" else "softmax_bf16.cc"
    kernel_path = REPO_ROOT / "allo" / "library" / "aie" / "kernels" / kernel_name
    top, source = softmax_wrapper_source(seq_len, kernel_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source, encoding="utf-8")
    return top, destination


@contextmanager
def flash_architecture_selector(device: str):
    """Select FA's legacy environment-based architecture without leaking it."""
    base.device_config(device)
    previous = os.environ.get("NPU2")
    os.environ["NPU2"] = "1" if device == "xdna2" else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NPU2", None)
        else:
            os.environ["NPU2"] = previous


def attention_reference(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute a stable float32 reference from the exact bf16 inputs."""
    q = Q.astype(np.float32)
    k = K.astype(np.float32)
    v = V.astype(np.float32)
    scores = (q @ k.T) * (1.0 / math.sqrt(HEAD_DIM))
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights @ v


def _build_baseline_modules(case: AttentionCase, project_root: Path):
    """Build the four parameterized baseline modules and return their buffers."""
    # Backend construction imports stay in the worker; Allo annotation symbols are
    # module-global so its source-based type resolver can find them.
    # pylint: disable=import-outside-toplevel,too-many-locals,used-before-assignment,not-callable
    import allo
    import allo.dataflow as df
    from allo.backend.aie.external_kernel import ExternalModule

    seq_len = case.seq_len
    config = base.device_config(case.device)
    device_type = config.allo_device_type(case.mapping_columns)
    tile_m = tile_n = 64

    score_pk = HEAD_DIM // 64
    score_pm = score_pn = seq_len // 64

    @df.region()
    def score_region(
        A: BF16_TYPE[seq_len, HEAD_DIM],
        B: BF16_TYPE[HEAD_DIM, seq_len],
        C: BF16_TYPE[seq_len, seq_len],
    ):
        pipe: Stream[BF16_TYPE[tile_m, tile_n], 2][score_pk - 1, score_pm, score_pn]

        @df.kernel(mapping=[score_pk, score_pm, score_pn], args=[A, B, C])
        def gemm(
            local_A: BF16_TYPE[seq_len, HEAD_DIM] @ GEMM_LAYOUT_A,
            local_B: BF16_TYPE[HEAD_DIM, seq_len] @ GEMM_LAYOUT_B,
            local_C: BF16_TYPE[seq_len, seq_len] @ GEMM_LAYOUT_C,
        ):
            pk, pm, pn = df.get_pid()
            C_in: BF16_TYPE[tile_m, tile_n]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: BF16_TYPE[tile_m, tile_n] = allo.add(
                allo.matmul(local_A, local_B), C_in
            )
            with allo.meta_if(pk < score_pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == score_pk - 1):
                local_C[:, :] = C_out

    score_module = df.build(
        score_region,
        target="aie",
        project=str(project_root / "score.prj"),
        mapping_primitives=gemm_mapping_primitives(
            score_pk,
            score_pm,
            score_pn,
            case.mapping_rows,
            case.mapping_columns,
        ),
        device_type=device_type,
    )

    scale_p0 = scale_p1 = seq_len // 64

    @df.region()
    def scale_region(
        C_in: BF16_TYPE[seq_len, seq_len], C_out: BF16_TYPE[seq_len, seq_len]
    ):
        @df.kernel(mapping=[scale_p0, scale_p1], args=[C_in, C_out])
        def core(
            local_C_in: BF16_TYPE[seq_len, seq_len] @ SCALE_LAYOUT,
            local_C_out: BF16_TYPE[seq_len, seq_len] @ SCALE_LAYOUT,
        ):
            local_C_out[:, :] = allo.mul(local_C_in, 0.125)

    scale_module = df.build(
        scale_region,
        target="aie",
        project=str(project_root / "scale.prj"),
        mapping_primitives=grid_mapping_primitives(
            "core",
            scale_p0,
            scale_p1,
            case.mapping_rows,
            case.mapping_columns,
        ),
        device_type=device_type,
    )

    softmax_top, softmax_path = write_softmax_wrapper(
        seq_len, case.device, project_root / f"softmax_bf16_{seq_len}.cc"
    )
    softmax_external = ExternalModule(
        top=softmax_top,
        impl_path=str(softmax_path),
        input_idx=[0],
        output_idx=[1],
    )
    softmax_p0 = seq_len // 4

    @df.region()
    def softmax_region(
        input_x: BF16_TYPE[seq_len, seq_len], output_x: BF16_TYPE[seq_len, seq_len]
    ):
        @df.kernel(mapping=[softmax_p0], args=[input_x, output_x])
        def core(
            local_input: BF16_TYPE[seq_len, seq_len] @ SOFTMAX_LAYOUT,
            local_output: BF16_TYPE[seq_len, seq_len] @ SOFTMAX_LAYOUT,
        ):
            softmax_external(local_input, local_output)

    softmax_module = df.build(
        softmax_region,
        target="aie",
        project=str(project_root / "softmax.prj"),
        mapping_primitives=linear_mapping_primitives(
            "core", softmax_p0, case.compute_slots
        ),
        device_type=device_type,
    )

    output_pk = seq_len // 64
    output_pm = seq_len // 64
    output_pn = HEAD_DIM // 64

    @df.region()
    def output_region(
        A: BF16_TYPE[seq_len, seq_len],
        B: BF16_TYPE[seq_len, HEAD_DIM],
        C: BF16_TYPE[seq_len, HEAD_DIM],
    ):
        pipe: Stream[BF16_TYPE[tile_m, tile_n], 2][output_pk - 1, output_pm, output_pn]

        @df.kernel(mapping=[output_pk, output_pm, output_pn], args=[A, B, C])
        def gemm(
            local_A: BF16_TYPE[seq_len, seq_len] @ GEMM_LAYOUT_A,
            local_B: BF16_TYPE[seq_len, HEAD_DIM] @ GEMM_LAYOUT_B,
            local_C: BF16_TYPE[seq_len, HEAD_DIM] @ GEMM_LAYOUT_C,
        ):
            pk, pm, pn = df.get_pid()
            C_in: BF16_TYPE[tile_m, tile_n]
            with allo.meta_if(pk > 0):
                C_in[:, :] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in[:, :] = 0
            C_out: BF16_TYPE[tile_m, tile_n] = allo.add(
                allo.matmul(local_A, local_B), C_in
            )
            with allo.meta_if(pk < output_pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == output_pk - 1):
                local_C[:, :] = C_out

    output_module = df.build(
        output_region,
        target="aie",
        project=str(project_root / "output.prj"),
        mapping_primitives=gemm_mapping_primitives(
            output_pk, output_pm, output_pn, case.compute_slots, 1
        ),
        device_type=device_type,
    )
    return score_module, scale_module, softmax_module, output_module


def _build_flash_module(case: AttentionCase, project_root: Path):
    """Build the fused FlashAttention module for a case."""
    # pylint: disable=import-outside-toplevel
    import allo.dataflow as df
    from allo.library.aie.modules.flash_attn import FA

    config = base.device_config(case.device)
    with flash_architecture_selector(case.device):
        top, mapping_primitives = FA(
            case.seq_len,
            HEAD_DIM,
            case.seq_len,
            Q_CHUNK_SIZE,
            KV_CHUNK_SIZE,
        )
    return df.build(
        top,
        target="aie",
        project=str(project_root / "flash.prj"),
        mapping_primitives=mapping_primitives,
        device_type=config.allo_device_type(case.mapping_columns),
    )


def worker_main(argv: Sequence[str]) -> int:
    """Build, validate, and benchmark one attention configuration."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=base.DEVICE_CHOICES, required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATION_ORDER, required=True)
    parser.add_argument("--seq-len", type=int, choices=DEFAULT_SEQ_LENS, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    case = AttentionCase(args.implementation, args.seq_len, args.device)
    config = base.device_config(case.device)

    # pylint: disable=import-outside-toplevel
    from ml_dtypes import bfloat16 as np_bfloat16

    kernel_dir = REPO_ROOT / "allo" / "library" / "aie" / "kernels"
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = str(kernel_dir.resolve()) + os.sep
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    os.environ["COALESCE_MORE"] = "1"
    os.environ["FORCE_UNROLL_INDEX"] = "0"
    os.environ["NPU2"] = config.npu2

    rng = np.random.default_rng(42 + case.seq_len)
    Q = rng.standard_normal((case.seq_len, HEAD_DIM)).astype(np_bfloat16)
    K = rng.standard_normal((case.seq_len, HEAD_DIM)).astype(np_bfloat16)
    V = rng.standard_normal((case.seq_len, HEAD_DIM)).astype(np_bfloat16)
    output = np.zeros((case.seq_len, HEAD_DIM), dtype=np_bfloat16)
    reference = attention_reference(Q, K, V)

    if case.implementation == "baseline":
        score_module, scale_module, softmax_module, output_module = (
            _build_baseline_modules(case, args.project_root)
        )
        scores = np.zeros((case.seq_len, case.seq_len), dtype=np_bfloat16)
        weights = np.zeros((case.seq_len, case.seq_len), dtype=np_bfloat16)

        def run_once():
            run_baseline_once(
                (
                    lambda: score_module(Q, K.T, scores),
                    lambda: scale_module(scores, scores),
                    lambda: softmax_module(scores, weights),
                    lambda: output_module(weights, V, output),
                )
            )

    else:
        flash_module = _build_flash_module(case, args.project_root)

        def run_once():
            flash_module(Q, K.T, V, output)

    run_once()
    try:
        np.testing.assert_allclose(
            output.astype(np.float32), reference, rtol=0.0, atol=5e-2
        )
    except AssertionError:
        if case.implementation != "baseline":
            raise
        traceback.print_exc()
        print(VALIDATION_FAILED, flush=True)
    else:
        print(VALIDATION_PASSED, flush=True)
    samples = measure_complete_attention(run_once, args.warmup, args.iterations)
    for sample in samples:
        print(f"{TIMING_PREFIX}{sample:.6f}", flush=True)
    return 0


def load_records(output_dir: Path) -> list[dict]:
    """Load all attention records from a result directory."""
    records = []
    for path in sorted((output_dir / "cases").glob("**/*.json")):
        with path.open(encoding="utf-8") as source:
            record = json.load(source)
        if record.get("schema_version") != base.SCHEMA_VERSION:
            raise ExperimentError(f"Unsupported result schema in {path}")
        if record.get("experiment") != EXPERIMENT_NAME:
            raise ExperimentError(f"Unexpected experiment record in {path}")
        records.append(record)
    return sorted(
        records,
        key=lambda item: (
            IMPLEMENTATION_ORDER.index(item["implementation"]),
            item["seq_len"],
        ),
    )


def describe_timings(values: Sequence[float], prefix: str) -> dict:
    """Return summary fields for one timing sequence."""
    fields = {
        f"{prefix}_mean_us": "",
        f"{prefix}_median_us": "",
        f"{prefix}_min_us": "",
        f"{prefix}_max_us": "",
        f"{prefix}_std_us": "",
    }
    if not values:
        return fields
    samples = [float(value) for value in values]
    return {
        f"{prefix}_mean_us": statistics.fmean(samples),
        f"{prefix}_median_us": statistics.median(samples),
        f"{prefix}_min_us": min(samples),
        f"{prefix}_max_us": max(samples),
        f"{prefix}_std_us": statistics.pstdev(samples),
    }


def process_results(output_dir: Path) -> tuple[Path, Path, Path]:
    """Regenerate raw, filtered, and summary CSV files."""
    output_dir = output_dir.resolve()
    records = load_records(output_dir)
    metadata_fields = [
        "case_id",
        "device",
        "target",
        "backend_target",
        "npu2",
        "implementation",
        "status",
        "validation",
        "timed_validation_failure",
        "dtype",
        "seq_len",
        "head_dim",
        "q_chunk_size",
        "kv_chunk_size",
        "mapping_rows",
        "mapping_columns",
        "compute_slots",
        "kernel_count",
        "timing_scope",
        "attention_timing_version",
        "npu_timing_scope",
        "npu_aggregation",
        "warmup",
        "iterations",
    ]
    timing_fields = metadata_fields + [
        "sample_index",
        "time_us",
        "npu_time_us",
        "extra_time_us",
        "is_outlier",
    ]
    component_summary_fields = [
        f"{population}_{component}_{stat}_us"
        for population in ("raw", "filtered")
        for component in ("npu", "extra")
        for stat in ("mean", "median", "min", "max", "std")
    ]
    summary_fields = metadata_fields + [
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
        "filtered_mean_us",
        "filtered_median_us",
        "filtered_min_us",
        "filtered_max_us",
        "filtered_std_us",
        *component_summary_fields,
        "elapsed_seconds",
        "error",
        "log_path",
    ]
    raw_rows = []
    filtered_rows = []
    summary_rows = []
    for record in records:
        timings = [float(value) for value in record.get("timings_us", [])]
        has_npu_timings = "npu_timings_us" in record
        npu_timings = [float(value) for value in record.get("npu_timings_us", [])]
        extra_timings = (
            derive_extra_timings(timings, npu_timings) if has_npu_timings else []
        )
        paired = has_npu_timings and len(timings) == len(npu_timings)
        filtered_data = base.tukey_filter(timings)
        completed = (
            record.get("status") == "success" and record.get("validation") == "passed"
        ) or is_timed_validation_failure(record)
        retained_indices = [
            index
            for index, retained in enumerate(filtered_data["mask"])
            if completed and retained
        ]
        for index, value in enumerate(timings):
            row = {field: record.get(field, "") for field in metadata_fields}
            row.update(
                {
                    "sample_index": index,
                    "time_us": value,
                    "npu_time_us": npu_timings[index] if paired else "",
                    "extra_time_us": extra_timings[index] if paired else "",
                    "is_outlier": not filtered_data["mask"][index],
                }
            )
            raw_rows.append(row)
            if index in retained_indices:
                filtered_rows.append(row.copy())

        filtered = [timings[index] for index in retained_indices]
        filtered_npu = (
            [npu_timings[index] for index in retained_indices] if paired else []
        )
        filtered_extra = (
            [extra_timings[index] for index in retained_indices] if paired else []
        )
        summary = {field: record.get(field, "") for field in metadata_fields}
        summary.update(
            {
                "raw_count": len(timings),
                "filtered_count": len(filtered),
                "outlier_count": len(timings) - len(filtered) if completed else "",
                "q1_us": filtered_data["q1"] if timings else "",
                "q3_us": filtered_data["q3"] if timings else "",
                "iqr_us": filtered_data["iqr"] if timings else "",
                "lower_bound_us": (filtered_data["lower_bound"] if timings else ""),
                "upper_bound_us": (filtered_data["upper_bound"] if timings else ""),
                "elapsed_seconds": record.get("elapsed_seconds", ""),
                "error": record.get("error", ""),
                "log_path": record.get("log_path", ""),
            }
        )
        summary.update(describe_timings(timings, "raw"))
        summary.update(describe_timings(filtered, "filtered"))
        summary.update(describe_timings(npu_timings, "raw_npu"))
        summary.update(describe_timings(filtered_npu, "filtered_npu"))
        summary.update(describe_timings(extra_timings, "raw_extra"))
        summary.update(describe_timings(filtered_extra, "filtered_extra"))
        summary_rows.append(summary)

    raw_path = output_dir / "raw_timings.csv"
    filtered_path = output_dir / "filtered_timings.csv"
    summary_path = output_dir / "summary.csv"
    base.atomic_write_csv(raw_path, timing_fields, raw_rows)
    base.atomic_write_csv(filtered_path, timing_fields, filtered_rows)
    base.atomic_write_csv(summary_path, summary_fields, summary_rows)
    return raw_path, filtered_path, summary_path


def preview_case(case: AttentionCase, args: argparse.Namespace) -> None:
    """Print one resolved dry-run configuration and execution disposition."""
    print(
        f"{case.case_id}: {case.device} {case.implementation} "
        f"N={case.seq_len} D={HEAD_DIM} mapping="
        f"{case.mapping_rows}x{case.mapping_columns} kernels={case.kernel_count}"
    )
    if case.infeasible:
        print(f"  infeasible: {INFEASIBLE_BASELINE_REASON}")
        return
    command = worker_command(
        case, args.warmup, args.iterations, work_path(case) / "projects"
    )
    print(f"  {base.render_command(command)}")


def run_experiments(args: argparse.Namespace) -> int:
    """Execute selected cases and regenerate aggregate tables."""
    cases = generate_cases(args.implementation, args.seq_lens, args.device)
    if args.dry_run:
        for case in cases:
            preview_case(case, args)
        print(f"Dry run: {len(cases)} configuration(s)")
        return 0

    if any(not case.infeasible for case in cases):
        base.check_environment(("allo",), base.DEFAULT_MLIR_AIE_ROOT)
    config = base.device_config(args.device)
    output_dir = (args.output_dir or default_output_dir(args.device)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["NPU2"] = config.npu2
    failures = 0
    interrupted = False
    for index, case in enumerate(cases, start=1):
        result_file = record_path(output_dir, case)
        signature = case_signature(case, args.warmup, args.iterations)
        case_work = work_path(case)
        if not args.rerun and is_resumable(result_file, signature):
            if not args.keep_builds:
                safe_remove_work(case_work)
            print(f"[{index}/{len(cases)}] SKIP {case.case_id} (completed)")
            continue

        if case.infeasible:
            print(
                f"[{index}/{len(cases)}] INFEASIBLE {case.case_id} "
                "(recorded without execution)"
            )
            safe_remove_work(case_work)
            case_log = log_path(output_dir, case)
            case_log.parent.mkdir(parents=True, exist_ok=True)
            case_log.write_text(INFEASIBLE_BASELINE_REASON + "\n", encoding="utf-8")
            record = new_record(case, args.warmup, args.iterations, output_dir)
            record.update(
                {
                    "status": "infeasible",
                    "validation": "not_run",
                    "error": INFEASIBLE_BASELINE_REASON,
                    "elapsed_seconds": 0.0,
                    "finished_at": base.utc_now(),
                }
            )
            base.atomic_write_json(result_file, record)
            continue

        print(f"[{index}/{len(cases)}] RUN  {case.case_id}")
        safe_remove_work(case_work)
        case_work.mkdir(parents=True, exist_ok=True)
        case_log = log_path(output_dir, case)
        case_log.parent.mkdir(parents=True, exist_ok=True)
        case_log.write_text("", encoding="utf-8")
        record = new_record(case, args.warmup, args.iterations, output_dir)
        command = worker_command(
            case, args.warmup, args.iterations, case_work / "projects"
        )
        record["commands"] = [base.render_command(command)]
        base.atomic_write_json(result_file, record)
        started = time.monotonic()
        try:
            return_code, output = base.run_command(
                command, REPO_ROOT, case_log, environment
            )
            if return_code != 0:
                raise ExperimentError("Attention worker failed; see the case log")
            validation_counts = {
                "passed": output.count(VALIDATION_PASSED),
                "failed": output.count(VALIDATION_FAILED),
            }
            if sum(validation_counts.values()) != 1:
                raise ExperimentError(
                    "Attention worker did not report exactly one validation result"
                )
            validation = next(
                name for name, count in validation_counts.items() if count == 1
            )
            if validation == "failed" and case.implementation != "baseline":
                raise ExperimentError(
                    "Only the unfused baseline may time a validation failure"
                )
            timings = parse_sample_timings(output, args.iterations)
            npu_timings = parse_npu_timings(output, case, args.warmup, args.iterations)
            derive_extra_timings(timings, npu_timings)
            if validation == "failed":
                record.update(
                    {
                        "status": "failed",
                        "validation": "failed",
                        "timed_validation_failure": True,
                        "timings_us": timings,
                        "npu_timings_us": npu_timings,
                        "error": (
                            "Baseline output validation failed; timings were recorded"
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
                        "npu_timings_us": npu_timings,
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
                {"status": "failed", "validation": "failed", "error": str(exc)}
            )
            print(f"FAILED {case.case_id}: {exc}", file=sys.stderr)
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            record["finished_at"] = base.utc_now()
            base.atomic_write_json(result_file, record)
            if not args.keep_builds:
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
    return 1 if failures else 0


def print_case_list(args: argparse.Namespace) -> int:
    """Print selected cases without requiring the hardware environment."""
    cases = generate_cases(args.implementation, args.seq_lens, args.device)
    for case in cases:
        disposition = "infeasible" if case.infeasible else "runnable"
        print(
            f"{case.case_id}: device={case.device} "
            f"mapping={case.mapping_rows}x{case.mapping_columns} "
            f"kernels={case.kernel_count} status={disposition}"
        )
    print(f"Total: {len(cases)} configuration(s)")
    return 0


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common case-selection options."""
    parser.add_argument(
        "--device", choices=base.DEVICE_CHOICES, default=base.DEFAULT_DEVICE
    )
    parser.add_argument(
        "--implementation",
        choices=(*IMPLEMENTATION_ORDER, "both"),
        default="both",
    )
    parser.add_argument(
        "--seq-len",
        dest="seq_lens",
        type=int,
        choices=DEFAULT_SEQ_LENS,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="one or more sequence lengths",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public attention experiment parser."""
    parser = argparse.ArgumentParser(
        description="Run and process end-to-end NPU attention experiments",
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
    run_parser.add_argument("--iterations", type=int, default=100)
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="result directory (device-specific when omitted)",
    )
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--keep-builds", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=run_experiments)

    list_parser = subparsers.add_parser(
        "list",
        help="list configurations without accessing hardware",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_selection_arguments(list_parser)
    list_parser.set_defaults(handler=print_case_list)

    process_parser = subparsers.add_parser(
        "process",
        help="regenerate CSV files from per-case JSON records",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        for path in process_results(output_dir):
            print(path)
        return 0

    process_parser.set_defaults(handler=process_handler)
    return parser


def validate_cli_arguments(args: argparse.Namespace) -> None:
    """Validate numeric arguments not expressible through argparse."""
    if args.command != "run":
        return
    if args.warmup < 0:
        raise ExperimentError("--warmup must be non-negative")
    if args.iterations <= 0:
        raise ExperimentError("--iterations must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the public CLI or internal worker."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "_worker":
        return worker_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        validate_cli_arguments(args)
        return args.handler(args)
    except (ExperimentError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
