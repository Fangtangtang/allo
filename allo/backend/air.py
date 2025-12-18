# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend helpers.

This module implements a minimal project-based runner used by unit tests.

Additionally, it provides :func:`convert` which translates an Allo-generated
MLIR module (stored in `<project>/allo.mlir`) into an AIR dialect module
(stored in `<project>/top.mlir`).

The conversion implemented here is intentionally minimal and targets the unit
tests under `tests/dataflow/air`.

Key points
----------
- We construct the surrounding `air.herd` using the MLIR parser (not the python
  op builders) to avoid fragile region terminator bindings.
- Inside the herd, we build DMA + compute by cloning ops and (for outputs)
  rewriting `memref.copy` into `air.dma_memcpy_nd`.

Supported patterns (unit tests)
------------------------------
- passthrough: memref.copy(arg0 -> arg1)
- vadd: linalg.add + memref.copy(tmp -> arg2)
- broadcast: linalg.fill/broadcast/add + memref.copy(tmp -> arg1)
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
from pathlib import Path

import filelock
import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import (
    Context,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    MemRefType,
    Module,
)

# aircc driver (compiler)
import air.compiler.aircc.main as aircc


def _resolve_project_dir(project: str) -> Path:
    """Resolve a project directory from a potentially relative name."""

    p = Path(project)
    if p.is_absolute():
        return p

    # 1) relative to CWD
    if p.exists():
        return p.resolve()

    # 2) relative to caller file directory
    for frame_info in inspect.stack()[1:]:
        f = frame_info.filename
        if not f:
            continue
        if Path(f).resolve() == Path(__file__).resolve():
            continue
        cand = (Path(f).resolve().parent / p).resolve()
        if cand.exists():
            return cand

    # 3) best-effort repo-relative guess
    repo_guess = Path(__file__).resolve().parents[2]  # .../workspace/allo
    cand = (repo_guess / "tests" / "dataflow" / "air" / p).resolve()
    if cand.exists():
        return cand

    raise FileNotFoundError(
        f"AIR project directory '{project}' not found. Tried CWD='{Path.cwd()}', caller-relative, and repo-relative paths."
    )


def _iter_external_kernel_sources(project_dir: Path) -> list[Path]:
    """Return top-level external kernel sources in the project directory."""

    srcs: list[Path] = []
    for pat in ("*.cc", "*.cpp", "*.cxx"):
        srcs.extend(sorted(project_dir.glob(pat)))
    return [p for p in srcs if p.is_file() and p.parent == project_dir]


def _detect_air_target_device() -> str:
    """Detect whether we target npu1 or npu2."""

    target_device = "npu2"
    try:
        xrtsmi = "/opt/xilinx/xrt/bin/xrt-smi"
        result = subprocess.run(
            [xrtsmi, "examine"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = result.stdout.decode("utf-8").split("\n")
        p = re.compile(r"[\|]?(\[.+:.+:.+\]).+\|(RyzenAI-(npu\d)|NPU (\w+))\W*\|")
        for l in out:
            m = p.match(l)
            if not m:
                continue
            model = "unknown"
            if m.group(3):
                model = str(m.group(3))
            if m.group(4):
                model = str(m.group(4))
            if model in ["npu1", "Phoenix"]:
                target_device = "npu1"
            elif model in ["npu4", "Strix"]:
                target_device = "npu2"
            break
    except Exception:
        pass
    return target_device


def _detect_aie_target_triple() -> str:
    return "aie2p" if os.environ.get("NPU2", "0") not in ("", "0") else "aie2"


def _compile_external_kernel(
    src: Path, project_dir: Path, *, verbose: bool = False
) -> Path:
    """Compile a single external kernel source to an object file."""

    peano = os.environ.get("PEANO_INSTALL_DIR")
    if not peano:
        raise RuntimeError(
            "PEANO_INSTALL_DIR is not set (required for external kernels)."
        )

    clangpp = Path(peano) / "bin" / "clang++"
    if not clangpp.exists():
        raise RuntimeError(f"Cannot find '{clangpp}'.")

    mlir_aie = os.environ.get("MLIR_AIE_INSTALL_DIR", "")
    if not mlir_aie:
        raise RuntimeError(
            "MLIR_AIE_INSTALL_DIR is not set (required for external kernel includes)."
        )

    mlir_aie_kernels = os.environ.get("MLIR_AIE_EXTERNAL_KERNEL_DIR", "")

    obj = project_dir / f"{src.name}.o"

    target = _detect_aie_target_triple()
    triple_flag = f"--target={target}-none-unknown-elf"

    cmd: list[str] = [
        str(clangpp),
        "-O2",
        "-std=c++20",
        triple_flag,
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-DNDEBUG",
        "-I",
        str(Path(mlir_aie) / "include"),
        "-I",
        str(project_dir),
    ]
    if mlir_aie_kernels:
        cmd += ["-I", mlir_aie_kernels]
    cmd += ["-c", str(src), "-o", str(obj)]

    if verbose:
        print("[allo.air]", " ".join(cmd))

    proc = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to compile external kernel.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )
    return obj


def _compile_external_kernels_if_needed(
    project_dir: Path, *, verbose: bool = False
) -> None:
    for src in _iter_external_kernel_sources(project_dir):
        _compile_external_kernel(src, project_dir, verbose=verbose)


def _compile_air_project(
    air_module: Module,
    *,
    project_dir: Path,
    build_dir: Path,
    xclbin_path: Path,
    insts_path: Path,
    kernel: str = "MLIR_AIE",
    verbose: bool = False,
) -> XRTCompileArtifact:
    """Compile AIR module to xclbin+insts, writing everything under project_dir."""

    build_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = build_dir / "aircc_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    target_device = _detect_air_target_device()

    aircc_options: list[str] = [
        "--device",
        target_device,
        "air.mlir",  # naming only; module passed in-memory
        "--tmpdir",
        str(tmpdir),
        "-o",
        str(xclbin_path),
        "-i",
        str(insts_path),
        "--output-format",
        "xclbin",
    ]

    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano:
        aircc_options += ["--peano", peano, "--no-xchesscc", "--no-xbridge"]
    else:
        aircc_options += ["--xchesscc", "--xbridge"]

    if verbose:
        aircc_options += ["-v"]

    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        aircc.run(air_module, aircc_options)
    finally:
        os.chdir(old_cwd)

    return XRTCompileArtifact(str(xclbin_path), kernel, str(insts_path))


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile and execute an AIR project."""

    project_dir = _resolve_project_dir(project)
    project_dir.mkdir(parents=True, exist_ok=True)

    top_mlir = project_dir / "top.mlir"
    if not top_mlir.exists():
        raise FileNotFoundError(f"'{top_mlir}' does not exist")

    np_args: list[np.ndarray] = []
    for a in args:
        if not isinstance(a, np.ndarray):
            raise TypeError(f"AIR backend expects numpy.ndarray args, got {type(a)}")
        np_args.append(a)

    build_dir = project_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    prj_lock_path = project_dir / ".allo_air_prj.lock"
    npu_lock_path = project_dir / ".allo_npu.lock"

    xclbin_path = build_dir / "final.xclbin"
    insts_path = build_dir / "insts.bin"

    with filelock.FileLock(str(prj_lock_path)):
        _compile_external_kernels_if_needed(project_dir)

        mlir_txt = top_mlir.read_text(encoding="utf-8")
        with Context() as ctx, Location.unknown():
            air_module = Module.parse(mlir_txt, ctx)

        artifact = _compile_air_project(
            air_module,
            project_dir=project_dir,
            build_dir=build_dir,
            xclbin_path=xclbin_path,
            insts_path=insts_path,
        )

        backend = XRTBackend(verbose=False)
        with filelock.FileLock(str(npu_lock_path)):
            invoker = backend.load(artifact)
            results = invoker(*np_args)
        backend.unload()

    for idx in output_idx:
        out = np_args[idx]
        got = np.asarray(results[idx]).reshape(out.shape)
        out[...] = got


# -----------------------------------------------------------------------------
# Allo MLIR -> AIR MLIR conversion
# -----------------------------------------------------------------------------


from air.dialects import air as air_d
from air.dialects import arith, func, linalg, memref


def _contiguous_strides(shape: list[int]) -> list[int]:
    stride = 1
    strides = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= int(shape[i])
    return strides


def _as_air_memref_type_with_space2(t) -> MemRefType:
    mt = MemRefType(t)
    i32 = IntegerType.get_signless(32)
    ms2 = IntegerAttr.get(i32, 2)
    return MemRefType.get(mt.shape, mt.element_type, layout=mt.layout, memory_space=ms2)


def _find_first_op(block, opname: str):
    for op in block.operations:
        if op.operation.name == opname:
            return op
    return None


def _infer_output_arg_positions(src_kernel) -> set[int]:
    """Infer output argument indices from `memref.copy` ops."""

    outs: set[int] = set()
    entry = src_kernel.entry_block
    args = list(src_kernel.arguments)
    for op in entry.operations:
        if op.operation.name != "memref.copy":
            continue
        dst = op.operation.operands[1]
        for i, a in enumerate(args):
            if dst == a:
                outs.add(i)
                break
    return outs


def _make_wrapper_mlir_for_top(arg_types: list[str]) -> str:
    """Create a minimal AIR module with @top containing an empty 1x1 herd."""

    args_sig = ", ".join([f"%arg{i}: {t}" for i, t in enumerate(arg_types)])
    herd_args = ", ".join([f"%ha{i}=%arg{i}" for i in range(len(arg_types))])
    herd_types = ", ".join(arg_types)

    return (
        "module {\n"
        f"  func.func @top({args_sig}) {{\n"
        "    %c1 = arith.constant 1 : index\n"
        f"    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) "
        f"args({herd_args}) : {herd_types} {{\n"
        "      air.herd_terminator\n"
        "    }\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _filter_memref_block_args(block_args):
    return [a for a in block_args if MemRefType.isinstance(a.type)]


def convert(project: str):
    """Translate `<project>/allo.mlir` to `<project>/top.mlir` (AIR dialect)."""

    project_dir = _resolve_project_dir(project)
    allo_mlir = project_dir / "allo.mlir"
    if not allo_mlir.exists():
        raise FileNotFoundError(f"'{allo_mlir}' does not exist")

    src_txt = allo_mlir.read_text(encoding="utf-8")

    with Context() as ctx, Location.unknown():
        src_mod = Module.parse(src_txt, ctx)

        # Locate @top and the single kernel (df.kernel).
        src_top = None
        src_kernel = None
        for op in src_mod.body.operations:
            if op.operation.name != "func.func":
                continue
            name = op.operation.attributes["sym_name"].value
            if name == "top":
                src_top = op
            if src_kernel is None and "df.kernel" in op.operation.attributes:
                src_kernel = op

        if src_top is None:
            raise RuntimeError("convert: cannot find func.func @top in allo.mlir")
        if src_kernel is None:
            raise RuntimeError(
                "convert: cannot find a kernel function (func.func with attribute 'df.kernel') in allo.mlir"
            )

        out_arg_pos = _infer_output_arg_positions(src_kernel)

        # Parse a wrapper AIR module that contains a well-formed herd terminator.
        top_arg_types = [str(a.type) for a in src_top.arguments]
        dst_mod = Module.parse(_make_wrapper_mlir_for_top(top_arg_types), ctx)

        dst_top = None
        for op in dst_mod.body.operations:
            if (
                op.operation.name == "func.func"
                and op.operation.attributes["sym_name"].value == "top"
            ):
                dst_top = op
                break
        if dst_top is None:
            raise RuntimeError("convert: internal error: wrapper @top not found")

        herd = _find_first_op(dst_top.entry_block, "air.herd")
        if herd is None:
            raise RuntimeError("convert: internal error: wrapper air.herd not found")

        herd_block = herd.body.blocks[0]
        herd_term = herd_block.operations[-1]

        # Choose memref-typed block args as kernel interface.
        herd_kernel_args = _filter_memref_block_args(herd_block.arguments)
        if len(herd_kernel_args) != len(src_kernel.arguments):
            raise RuntimeError(
                "convert: cannot match kernel arguments to herd block arguments. "
                f"kernel_args={len(src_kernel.arguments)} herd_memref_args={len(herd_kernel_args)}"
            )

        i32_t = IntegerType.get_signless(32)
        ms2 = IntegerAttr.get(i32_t, 2)
        idx_t = IndexType.get()

        # Insert before herd terminator.
        with InsertionPoint(herd_term):
            c0 = arith.ConstantOp(idx_t, IntegerAttr.get(idx_t, 0))

            value_map = {}
            allocs_to_dealloc: list = []

            # Precompute dst slice descriptors for each output arg.
            out_slice_desc: dict[int, tuple[list, list, list]] = {}
            for out_i in out_arg_pos:
                mt = MemRefType(herd_kernel_args[out_i].type)
                shape = [int(d) for d in mt.shape]
                strides = _contiguous_strides(shape)
                size_consts = [
                    arith.ConstantOp(idx_t, IntegerAttr.get(idx_t, int(d))).result
                    for d in shape
                ]
                stride_consts = [
                    arith.ConstantOp(idx_t, IntegerAttr.get(idx_t, int(s))).result
                    for s in strides
                ]
                offset_consts = [c0.result for _ in shape]
                out_slice_desc[out_i] = (offset_consts, size_consts, stride_consts)

            # Allocate interface buffers.
            for arg_i, (src_arg, herd_arg) in enumerate(
                zip(src_kernel.arguments, herd_kernel_args)
            ):
                src_mt = MemRefType(herd_arg.type)
                if any(int(d) < 0 for d in src_mt.shape):
                    raise RuntimeError(
                        f"convert: dynamic shapes not supported in unit tests (got {src_mt})"
                    )

                if arg_i in out_arg_pos:
                    # Outputs are written via rewritten DMA; map arg to the global memref.
                    value_map[src_arg] = herd_arg
                    continue

                # Inputs: DMA global->local.
                local_t = MemRefType.get(
                    src_mt.shape,
                    src_mt.element_type,
                    layout=src_mt.layout,
                    memory_space=ms2,
                )
                local_alloc = memref.AllocOp(local_t, [], [])
                allocs_to_dealloc.append(local_alloc.result)

                shape = [int(d) for d in src_mt.shape]
                strides = _contiguous_strides(shape)
                size_consts = [
                    arith.ConstantOp(idx_t, IntegerAttr.get(idx_t, int(d))).result
                    for d in shape
                ]
                stride_consts = [
                    arith.ConstantOp(idx_t, IntegerAttr.get(idx_t, int(s))).result
                    for s in strides
                ]
                offset_consts = [c0.result for _ in shape]

                air_d.DmaMemcpyNdOp(
                    None,
                    [],
                    local_alloc.result,
                    [],
                    [],
                    [],
                    herd_arg,
                    offset_consts,
                    size_consts,
                    stride_consts,
                )

                value_map[src_arg] = local_alloc.result

            # Clone/rewrite kernel operations into herd body.
            src_args = list(src_kernel.arguments)
            for op in src_kernel.entry_block.operations:
                opname = op.operation.name
                if opname == "func.return":
                    continue

                # Convert allocs to L2.
                if opname == "memref.alloc":
                    old_res = op.operation.results[0]
                    new_t = _as_air_memref_type_with_space2(old_res.type)
                    new_alloc = memref.AllocOp(new_t, [], [])
                    value_map[old_res] = new_alloc.result
                    allocs_to_dealloc.append(new_alloc.result)
                    continue

                # Rewrite output copies to DMA.
                if opname == "memref.copy":
                    src_v = op.operation.operands[0]
                    dst_v = op.operation.operands[1]
                    out_idx = None
                    for i, a in enumerate(src_args):
                        if dst_v == a:
                            out_idx = i
                            break
                    if out_idx is not None and out_idx in out_arg_pos:
                        # dst is an output argument: emit DMA to global output.
                        dma_src = value_map.get(src_v, src_v)
                        dst_global = herd_kernel_args[out_idx]
                        offs, sizes, strides = out_slice_desc[out_idx]
                        air_d.DmaMemcpyNdOp(
                            None,
                            [],
                            dst_global,
                            offs,
                            sizes,
                            strides,
                            dma_src,
                            [],
                            [],
                            [],
                        )
                        continue

                # Default: clone and remap operands.
                new_operation = op.operation.clone()
                for i in range(len(new_operation.operands)):
                    v = new_operation.operands[i]
                    if v in value_map:
                        new_operation.operands[i] = value_map[v]
                for old_res, new_res in zip(
                    op.operation.results, new_operation.results
                ):
                    value_map[old_res] = new_res

            for v in reversed(allocs_to_dealloc):
                memref.DeallocOp(v)

        # Pretty-print in custom assembly form.
        top_mlir_txt = dst_mod.operation.get_asm(print_generic_op_form=False)
        (project_dir / "top.mlir").write_text(top_mlir_txt, encoding="utf-8")
