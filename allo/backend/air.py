# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend helpers.

This module implements a minimal project-based runner used by unit tests.

Requirements satisfied:
- Read AIR dialect MLIR from `<project>/top.mlir`.
- Compile using mlir-air's compiler driver (aircc) and run via XRT.
- Keep *all* intermediate files under the given project directory.
- Support external kernels (`*.cc/*.cpp`) compiled with `$PEANO_INSTALL_DIR/bin/clang++`.

The unit tests call :func:`_call_prj` directly.

Additionally, this file provides :func:`convert` which translates an Allo-generated
MLIR module (stored in `<project>/allo.mlir`) into an AIR dialect module
(stored in `<project>/top.mlir`).

NOTE: The conversion implemented here is intentionally minimal: it targets the
vector tests in `tests/dataflow/air/`.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator

import filelock
import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import (
    Context,
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
    """Detect whether we target npu1 or npu2.

    Keep it lightweight: follow mlir-air's own XRTBackend heuristic by probing xrt-smi.
    If anything fails, default to npu2.
    """

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
    """Return AIE target triple prefix (aie2 or aie2p)."""

    # This is sufficient for our tests.
    return "aie2p" if os.environ.get("NPU2", "0") not in ("", "0") else "aie2"


def _compile_external_kernel(
    src: Path, project_dir: Path, *, verbose: bool = False
) -> Path:
    """Compile a single external kernel source to an object file.

    Object naming is important: mlir-air expects `passThrough.cc.o` if the IR
    says `link_with = "passThrough.cc.o"`.
    """

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
    srcs = _iter_external_kernel_sources(project_dir)
    if not srcs:
        return
    for src in srcs:
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

    # Absolute tmpdir to satisfy sandboxing (relative paths can be blocked by LD_PRELOAD).
    tmpdir = build_dir / "aircc_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    target_device = _detect_air_target_device()

    aircc_options: list[str] = [
        "--device",
        target_device,
        "air.mlir",  # only used for naming; the module is passed in-memory
        "--tmpdir",
        str(tmpdir),
        "-o",
        str(xclbin_path),
        "-i",
        str(insts_path),
        "--output-format",
        "xclbin",
    ]

    # Use peano toolchain if available (common in RyzenAI setups)
    peano = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano:
        aircc_options += ["--peano", peano, "--no-xchesscc", "--no-xbridge"]
    else:
        aircc_options += ["--xchesscc", "--xbridge"]

    if verbose:
        aircc_options += ["-v"]

    # Run aircc from the *project* directory so relative `link_with = "*.o"`
    # resolves to the object files we compiled into project_dir.
    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        aircc.run(air_module, aircc_options)
    finally:
        os.chdir(old_cwd)

    return XRTCompileArtifact(str(xclbin_path), kernel, str(insts_path))


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile and execute an AIR project.

    `args` are numpy arrays, including both inputs and pre-allocated outputs.
    For every index listed in `output_idx`, this function overwrites `args[idx]`
    in-place with the produced output.

    All intermediate files are written under the project directory.
    """

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

    # Locks live in the project directory (not /tmp).
    prj_lock_path = project_dir / ".allo_air_prj.lock"
    npu_lock_path = project_dir / ".allo_npu.lock"

    xclbin_path = build_dir / "final.xclbin"
    insts_path = build_dir / "insts.bin"

    with filelock.FileLock(str(prj_lock_path)):
        # 1) Compile external kernels if any.
        _compile_external_kernels_if_needed(project_dir)

        # 2) Parse MLIR.
        mlir_txt = top_mlir.read_text(encoding="utf-8")
        with Context() as ctx, Location.unknown():
            air_module = Module.parse(mlir_txt, ctx)

        # 3) Compile AIR->xclbin/insts (all artifacts under project_dir).
        artifact = _compile_air_project(
            air_module,
            project_dir=project_dir,
            build_dir=build_dir,
            xclbin_path=xclbin_path,
            insts_path=insts_path,
        )

        # 4) Execute using XRT.
        backend = XRTBackend(verbose=False)
        with filelock.FileLock(str(npu_lock_path)):
            invoker = backend.load(artifact)
            results = invoker(*np_args)
        backend.unload()

    # 5) Copy results back.
    for idx in output_idx:
        out = np_args[idx]
        got = np.asarray(results[idx]).reshape(out.shape)
        out[...] = got


# -----------------------------------------------------------------------------
# Allo MLIR -> AIR MLIR conversion
# -----------------------------------------------------------------------------


def _walk_ops(op) -> Iterator:
    """Yield `Operation`s in preorder starting at `op` (inclusive)."""

    operation = op.operation if hasattr(op, "operation") else op
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_ops(child)


def _has_air_ops(module: Module) -> bool:
    return any(str(op.name).startswith("air.") for op in _walk_ops(module.operation))


def _memref_with_memory_space(t, memory_space: int):
    mt = MemRefType(t)
    i32 = IntegerType.get_signless(32)
    ms_attr = IntegerAttr.get(i32, memory_space)
    return MemRefType.get(mt.shape, mt.element_type, memory_space=ms_attr)


def convert(project: str):
    """Translate `<project>/allo.mlir` into `<project>/top.mlir`.

    Supported inputs (as used by tests):
    - Already-AIR IR: pass-through.
    - Linalg-only kernels for:
      * vector add
      * broadcast-add-one
      * passthrough

    All intermediate files are written only under the given project directory.
    """

    from air.dialects import air as air_d
    from air.dialects import arith, func, linalg, memref
    from air.ir import FunctionType

    project_dir = _resolve_project_dir(project)
    project_dir.mkdir(parents=True, exist_ok=True)

    allo_path = project_dir / "allo.mlir"
    if not allo_path.exists():
        raise FileNotFoundError(f"'{allo_path}' does not exist")

    content = allo_path.read_text(encoding="utf-8")

    with Context() as ctx, Location.unknown():
        in_module = Module.parse(content, ctx)

        # 0) If AIR ops already present, just write it out.
        if _has_air_ops(in_module):
            (project_dir / "top.mlir").write_text(str(in_module), encoding="utf-8")
            return

        # 1) Find the 'top' and a single df.kernel function.
        top_func = None
        kernel_func = None
        for op in in_module.body.operations:
            if isinstance(op, func.FuncOp):
                if op.name.value == "top":
                    top_func = op
                if "df.kernel" in op.attributes:
                    kernel_func = op

        if top_func is None:
            for op in in_module.body.operations:
                if isinstance(op, func.FuncOp):
                    top_func = op
                    break
        if top_func is None:
            raise RuntimeError("No func.func found in input allo.mlir")

        if kernel_func is None:
            kernel_func = top_func

        # 2) Infer which arguments are outputs by looking at memref.copy into block arguments.
        arg_is_output = [False] * len(kernel_func.arguments)
        arg_is_input = [False] * len(kernel_func.arguments)
        for op in kernel_func.entry_block.operations:
            if op.operation.name != "memref.copy":
                continue
            src, dst = op.operands[0], op.operands[1]
            for i, barg in enumerate(kernel_func.arguments):
                if dst is barg:
                    arg_is_output[i] = True
                if src is barg:
                    arg_is_input[i] = True

        if not any(arg_is_output) and len(arg_is_output) > 0:
            # Conservative fallback: last argument is output.
            arg_is_output[-1] = True
            for i in range(len(arg_is_input) - 1):
                arg_is_input[i] = True

        # 3) Detect which kernel pattern we need.
        op_names = [o.operation.name for o in _walk_ops(kernel_func)]
        is_broadcast = any(n == "linalg.broadcast" for n in op_names)
        is_add = any(n == "linalg.add" for n in op_names)
        is_passthrough = (not is_broadcast) and (not is_add)

        # 4) Build new AIR module.
        out_module = Module.create()
        with InsertionPoint(out_module.body):
            arg_types = [a.type for a in top_func.arguments]
            out_func = func.FuncOp("top", FunctionType.get(arg_types, []))
            out_entry = out_func.add_entry_block()

            with InsertionPoint(out_entry):
                top_args = list(out_entry.arguments)

                # Create herd. Using the ext wrapper ensures the region block args are set up.
                herd_op = air_d.Herd(name="herd_0", sizes=[1, 1], operands=top_args)
                herd_block = herd_op.body.blocks[0]
                herd_args = list(herd_block.arguments[4:])  # skip tile + herd size args

                with InsertionPoint(herd_block):
                    allocated = []  # Values we must dealloc.

                    c0 = arith.ConstantOp.create_index(0)
                    c1 = arith.ConstantOp.create_index(1)

                    # Allocate local buffers for each relevant argument.
                    local_bufs = [None] * len(herd_args)
                    for i, g in enumerate(herd_args):
                        if not isinstance(g.type, MemRefType):
                            local_bufs[i] = g
                            continue
                        if arg_is_input[i] or (
                            arg_is_output[i] and (not is_passthrough)
                        ):
                            local_t = _memref_with_memory_space(g.type, 2)
                            v = memref.AllocOp(local_t, [], []).result
                            local_bufs[i] = v
                            allocated.append(v)
                        else:
                            # For passthrough outputs we will reuse an input buffer.
                            local_bufs[i] = g

                    # DMAs for inputs: global -> local.
                    for i, g in enumerate(herd_args):
                        if not arg_is_input[i]:
                            continue
                        mt = MemRefType(g.type)
                        if len(mt.shape) != 1:
                            raise RuntimeError(
                                f"Only 1D memrefs are supported in test conversion, got {mt}"
                            )
                        M = int(mt.shape[0])
                        cM = arith.ConstantOp.create_index(M)
                        air_d.dma_memcpy_nd(
                            dst=local_bufs[i],
                            src=g,
                            dst_offsets=[],
                            dst_sizes=[],
                            dst_strides=[],
                            src_offsets=[c0],
                            src_sizes=[cM],
                            src_strides=[c1],
                        )

                    # Compute.
                    if is_broadcast:
                        # Pattern: B[:] = A + 1
                        mt0 = MemRefType(herd_args[0].type)
                        elem_t = mt0.element_type

                        one = arith.ConstantOp(elem_t, 1).result

                        scalar_t = _memref_with_memory_space(
                            MemRefType.get([], elem_t), 2
                        )
                        scalar = memref.AllocOp(scalar_t, [], []).result
                        allocated.append(scalar)
                        linalg.fill(one, outs=[scalar])

                        vec_t = _memref_with_memory_space(herd_args[0].type, 2)
                        bc_vec = memref.AllocOp(vec_t, [], []).result
                        allocated.append(bc_vec)
                        linalg.broadcast(scalar, outs=[bc_vec], dimensions=[0])

                        out_idx = next(i for i, f in enumerate(arg_is_output) if f)
                        linalg.add(local_bufs[0], bc_vec, outs=[local_bufs[out_idx]])

                    elif is_add:
                        # Pattern: C[:] = A + B
                        out_idx = next(i for i, f in enumerate(arg_is_output) if f)
                        in0 = local_bufs[0]
                        in1_idx = next(
                            i for i in range(1, len(local_bufs)) if arg_is_input[i]
                        )
                        in1 = local_bufs[in1_idx]
                        linalg.add(in0, in1, outs=[local_bufs[out_idx]])

                    else:
                        # Passthrough: B[:] = A
                        out_idx = next(i for i, f in enumerate(arg_is_output) if f)
                        in_idx = next(i for i, f in enumerate(arg_is_input) if f)
                        local_bufs[out_idx] = local_bufs[in_idx]

                    # DMAs for outputs: local -> global.
                    for i, g in enumerate(herd_args):
                        if not arg_is_output[i]:
                            continue
                        mt = MemRefType(g.type)
                        if len(mt.shape) != 1:
                            raise RuntimeError(
                                f"Only 1D memrefs are supported in test conversion, got {mt}"
                            )
                        M = int(mt.shape[0])
                        cM = arith.ConstantOp.create_index(M)
                        air_d.dma_memcpy_nd(
                            dst=g,
                            src=local_bufs[i],
                            dst_offsets=[c0],
                            dst_sizes=[cM],
                            dst_strides=[c1],
                            src_offsets=[],
                            src_sizes=[],
                            src_strides=[],
                        )

                    # Dealloc everything we allocated.
                    freed = set()
                    for v in allocated:
                        if v in freed:
                            continue
                        freed.add(v)
                        memref.DeallocOp(v)

                    air_d.HerdTerminatorOp()

                func.ReturnOp([])

        (project_dir / "top.mlir").write_text(str(out_module), encoding="utf-8")
