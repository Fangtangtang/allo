# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DRAM <-> NPU effective-bandwidth micro-benchmark for XDNA1.
# Reuses the canonical GEMM(...) factory unchanged. The MAC microkernel is
# swapped via ALLO_EXTERNAL_KERNEL_DIR -> dram_bw_kernels/mm.cc, whose helper
# bodies write zeros instead of multiply-accumulating. DMA descriptors are
# byte-identical to a real GEMM run, so the measured time reflects DDR<->NPU
# transfer at the same access pattern.

import argparse
import csv
import os
import re
import subprocess
import sys

import numpy as np
from ml_dtypes import bfloat16 as np_bfloat16

import allo.dataflow as df
from allo.backend.aie import is_available
from allo.ir.types import int8, int16, bfloat16
from allo.library.aie.modules.gemm import GEMM


DTYPES = {
    "int8": (int8, np.int8, 1),
    "int16": (int16, np.int16, 2),
    "bf16": (bfloat16, np_bfloat16, 2),
}

DEVICE_GRID = {
    "npu1_4col": (4, 4),
    "npu1_2col": (4, 2),
    "npu1_1col": (4, 1),
}

TIME_RE = re.compile(r"Avg NPU execution time:\s*([0-9]+(?:\.[0-9]+)?)us")


def dram_bytes(M, N, K, m, n, kt, row_num, col_num, elem_bytes):
    # User-chosen formula: counts grid broadcast as the only on-chip reuse.
    # A is broadcast across col_num cores -> M*N*K/(n*col_num) bytes
    # B is broadcast across row_num cores -> M*N*K/(m*row_num) bytes
    # C is written once               -> M*N bytes
    a_bytes = (M * N * K) // (n * col_num)
    b_bytes = (M * N * K) // (m * row_num)
    c_bytes = M * N
    return (a_bytes + b_bytes + c_bytes) * elem_bytes


def make_inputs(M, N, K, np_dtype):
    if np_dtype is np_bfloat16:
        A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
        B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
        C = np.zeros((M, N)).astype(np_bfloat16)
    else:
        info = np.iinfo(np_dtype)
        A = np.random.randint(-4, 4, (M, K)).astype(np_dtype)
        B = np.random.randint(-4, 4, (K, N)).astype(np_dtype)
        C = np.zeros((M, N)).astype(np_dtype)
        _ = info
    return A, B, C


def run_one(cfg, kernel_dir, warmup, num_iters, base_project_dir):
    M, N, K = cfg["M"], cfg["N"], cfg["K"]
    m, n, kt = cfg["m"], cfg["n"], cfg["k"]
    dtype_name = cfg["dtype"]
    device = cfg["device"]
    row_num, col_num = DEVICE_GRID[device]

    if device == "npu2":
        os.environ["NPU2"] = "1"
    else:
        os.environ.pop("NPU2", None)
    os.environ["ENABLE_AGGRESSIVE_PORT_UTILIZATION_PATCH"] = "1"
    os.environ["ALLO_EXTERNAL_KERNEL_DIR"] = kernel_dir

    TyI_allo, np_dtype, elem_bytes = DTYPES[dtype_name]
    TyO_allo = TyI_allo

    Pm, Pn, Pk = M // m, N // n, K // kt
    top, mapping_primitives = GEMM(
        M, N, K, Pm, Pn, Pk, TyI_allo, TyO_allo, col_num=col_num, row_num=row_num
    )

    project = os.path.join(
        base_project_dir,
        f"bw_{device}_{dtype_name}_M{M}_N{N}_K{K}_m{m}_n{n}_k{kt}.prj",
    )

    mod = df.build(
        top,
        project=project,
        target="aie",
        mapping_primitives=mapping_primitives,
        profile=True,
        warmup=warmup,
        num_iters=num_iters,
    )

    A, B, C = make_inputs(M, N, K, np_dtype)
    mod(A, B, C)

    # Re-run the freshly built binary to capture timing on stdout. The first
    # mod(...) call streams stdout uncaptured; this second invocation reuses
    # the project dir and the input*.data files just written.
    cmd = (
        f"./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE "
        f"--trace_sz 0 -p true --warmup {warmup} --test_iter {num_iters}"
    )
    result = subprocess.run(
        cmd,
        cwd=project,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    match = TIME_RE.search(result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not find 'Avg NPU execution time' in stdout:\n{result.stdout}"
        )
    time_us = float(match.group(1))

    bytes_xfer = dram_bytes(M, N, K, m, n, kt, row_num, col_num, elem_bytes)
    gb_per_s = bytes_xfer / time_us / 1e3  # bytes / us / 1e3 = GB/s

    return {
        "M": M,
        "N": N,
        "K": K,
        "m": m,
        "n": n,
        "k": kt,
        "dtype": dtype_name,
        "device": device,
        "row_num": row_num,
        "col_num": col_num,
        "bytes": bytes_xfer,
        "time_us": time_us,
        "GB_per_s": gb_per_s,
    }


def headline_cfg():
    return {
        "M": 2048,
        "N": 2048,
        "K": 2048,
        "m": 64,
        "n": 64,
        "k": 64,
        "dtype": "int8",
        "device": "npu1_4col",
    }


def sweep_configs():
    configs = []
    # Tile-size sweep at headline shape, int8, npu1_4col
    # for t in (32, 64, 128):
    #     configs.append(
    #         {"M": 2048, "N": 2048, "K": 2048, "m": t, "n": t, "k": t,
    #          "dtype": "int8", "device": "npu1_4col"}
    #     )
    # Dtype sweep at headline shape, npu1_4col, m=n=k=64
    # for d in ("int8", "int16", "bf16"):
    #     configs.append(
    #         {"M": 2048, "N": 2048, "K": 2048, "m": 64, "n": 64, "k": 64,
    #          "dtype": d, "device": "npu1_4col"}
    #     )
    # # Shim-column sweep at headline shape, int8, m=n=k=64
    # for dev in ("npu1_1col", "npu1_2col", "npu1_4col"):
    #     configs.append(
    #         {"M": 2048, "N": 2048, "K": 2048, "m": 64, "n": 64, "k": 64,
    #          "dtype": "int8", "device": dev}
    #     )
    # Problem-size sensitivity at int8, npu1_4col, m=n=k=64
    for d in ("int8", "int16", "bf16"):
        for s0 in (512, 1024, 2048):
            for s1 in (512, 1024, 2048):
                for s2 in (512, 1024, 2048):
                    configs.append(
                        {"M": s0, "N": s1, "K": s2, "m": 64, "n": 64, "k": 64,
                        "dtype": d, "device": "npu1_4col"}
                    )
    # Dedupe (headline appears in multiple slices)
    seen = set()
    unique = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headline-only", action="store_true",
                        help="Only run the headline config; skip sweep.")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--out", default=None,
                        help="CSV output path (defaults next to script).")
    args = parser.parse_args()

    if not is_available():
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    kernel_dir = os.path.join(here, "dram_bw_kernels")
    if not os.path.isfile(os.path.join(kernel_dir, "mm.cc")):
        raise RuntimeError(f"Missing stub kernel at {kernel_dir}/mm.cc")
    base_project_dir = os.path.join(here, "dram_bw_projects")
    os.makedirs(base_project_dir, exist_ok=True)

    out_path = args.out or os.path.join(here, "dram_gemm_microbench.csv")
    fieldnames = ["M", "N", "K", "m", "n", "k", "dtype", "device",
                  "row_num", "col_num", "bytes", "time_us", "GB_per_s"]

    # Write header once, truncating any previous file.
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def append_row(res):
        # Open-append-close per row so each completed config is durably on
        # disk before the next build starts. External readers (tail, editor)
        # see results as soon as they're written.
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(res)
            f.flush()
            os.fsync(f.fileno())

    n_rows = 0
    print("=== Headline config ===")
    head = headline_cfg()
    res = run_one(head, kernel_dir, args.warmup, args.num_iters, base_project_dir)
    print(f"  {res}")
    print(f"  -> {res['GB_per_s']:.2f} GB/s at "
          f"{res['time_us']:.1f} us, {res['bytes']/1e6:.1f} MB/iter")
    append_row(res)
    n_rows += 1

    if not args.headline_only:
        print("=== Sweep ===")
        for cfg in sweep_configs():
            if cfg == head:
                continue
            try:
                res = run_one(cfg, kernel_dir, args.warmup, args.num_iters,
                              base_project_dir)
                print(f"  {res['device']} {res['dtype']} "
                      f"M={res['M']} m={res['m']}: "
                      f"{res['GB_per_s']:.2f} GB/s ({res['time_us']:.1f} us)")
                append_row(res)
                n_rows += 1
            except Exception as e:
                print(f"  FAILED {cfg}: {e}", file=sys.stderr)

    print(f"Wrote {n_rows} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
