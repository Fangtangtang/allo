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

NOTE: This conversion is intentionally minimal and currently targets the
single-kernel vector addition / broadcast / passthrough style programs used in
unit tests.
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
from air.ir import Context, Location, Module

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
        # best-effort fallback
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
        # fall back to defaults
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


def _append_air_dialect_registry(ctx: Context) -> None:
    """Append AIR's DialectRegistry to the given context."""

    from air._mlir_libs import get_dialect_registry

    ctx.append_dialect_registry(get_dialect_registry())


def _memref_type_with_memory_space_2(t):
    """Return a MemRefType identical to `t` but in memory space `2 : i32`."""

    from air.ir import IntegerAttr, IntegerType, MemRefType

    mt = MemRefType(t)
    if mt.memory_space is not None:
        return mt
    ms2 = IntegerAttr.get(IntegerType.get_signless(32), 2)
    return MemRefType.get(mt.shape, mt.element_type, mt.layout, memory_space=ms2)


def _block_arg_number(v) -> int | None:
    """Return arg number if `v` is a block argument, else None."""

    try:
        from air.ir import BlockArgument

        if isinstance(v, BlockArgument):
            return int(v.arg_number)
    except Exception:
        pass
    return None


def convert(project: str):
    """Convert `<project>/allo.mlir` (Allo-generated MLIR) into AIR dialect `top.mlir`.

    This is a *dialect-to-dialect* transformation at the MLIR level.

    For the current unit tests, the input module has:
      - exactly one `df.kernel` function (`@core_0`) implementing the compute
      - a `@top` function that is essentially a signature carrier

    The output AIR module wraps the kernel in an `air.herd` and inserts DMA copies
    between external memrefs (top args) and L1 memrefs (memory space 2).
    """

    project_dir = _resolve_project_dir(project)
    allo_mlir_path = project_dir / "allo.mlir"
    if not allo_mlir_path.exists():
        raise FileNotFoundError(f"'{allo_mlir_path}' does not exist")

    content = allo_mlir_path.read_text(encoding="utf-8")

    # Parse the input module using the AIR python bindings so we can clone ops and
    # create AIR ops in the same context.
    with Context() as ctx, Location.unknown():
        _append_air_dialect_registry(ctx)
        # Make sure builtin dialects needed by the input are available.
        ctx.load_all_available_dialects()
        src_module = Module.parse(content, ctx)

        from air.dialects import func as func_d

        core_func = None
        top_func = None
        for op in src_module.body.operations:
            if not isinstance(op, func_d.FuncOp):
                continue
            if "df.kernel" in op.attributes:
                core_func = op
            else:
                if op.name.value == "top":
                    top_func = op
                elif top_func is None:
                    top_func = op

        if core_func is None:
            raise ValueError(
                "Cannot find a kernel function (expected a func.func with 'df.kernel' attribute)."
            )
        if top_func is None:
            raise ValueError("Cannot find a top-level function to use as entry.")

        # Detect outputs by looking for `memref.copy` whose dst is a block argument.
        output_arg_numbers: set[int] = set()
        for op in core_func.entry_block.operations:
            if op.operation.name != "memref.copy":
                continue
            dst = op.operands[1]
            an = _block_arg_number(dst)
            if an is not None:
                output_arg_numbers.add(an)

        # Build destination module.
        dst_module = Module.create()

        from air.dialects import air as air_d
        from air.dialects import arith, memref
        from air.dialects import func as air_func
        from air.ir import InsertionPoint, MemRefType

        with InsertionPoint(dst_module.body):
            entry = air_func.FuncOp(top_func.name.value, top_func.type)
            entry_block = entry.add_entry_block()

            with InsertionPoint(entry_block):
                c1 = arith.ConstantOp.create_index(1)
                herd_op = air_d.Herd(
                    name="herd_0",
                    sizes=[c1, c1],
                    operands=list(entry_block.arguments),
                )

                herd_block = herd_op.body.blocks[0]
                herd_operands = herd_block.arguments[4:]

                # Map from old SSA values (kernel function args/results) to new SSA values.
                value_map = {}
                allocs_to_dealloc: list = []

                with InsertionPoint(herd_block):
                    # Constants used inside the herd must be defined inside the herd region.
                    c0 = arith.ConstantOp.create_index(0)
                    c1s = arith.ConstantOp.create_index(1)

                    # 1) Allocate L1 buffers for every kernel argument.
                    for i, barg in enumerate(core_func.entry_block.arguments):
                        if not isinstance(barg.type, MemRefType):
                            raise ValueError(
                                f"Kernel argument {i} must be a memref for AIR conversion, got {barg.type}"
                            )
                        local_t = _memref_type_with_memory_space_2(barg.type)
                        local = memref.AllocOp(local_t, [], []).result
                        allocs_to_dealloc.append(local)
                        value_map[barg] = local

                        # Input DMA (skip outputs).
                        if i not in output_arg_numbers:
                            mt = MemRefType(barg.type)
                            rank = mt.rank
                            offsets = [c0] * rank
                            sizes = [arith.ConstantOp.create_index(int(d)) for d in mt.shape]
                            strides = [c1s] * rank

                            air_d.DmaMemcpyNd(
                                dst=local,
                                src=herd_operands[i],
                                dst_offsets=[],
                                dst_sizes=[],
                                dst_strides=[],
                                src_offsets=offsets,
                                src_sizes=sizes,
                                src_strides=strides,
                            )

                    # 2) Clone/translate kernel body operations.
                    for op in core_func.entry_block.operations:
                        opname = op.operation.name
                        if opname == "func.return":
                            continue

                        if opname == "memref.alloc":
                            # Allocate in L1 (memory space 2).
                            t = _memref_type_with_memory_space_2(op.result.type)
                            new_alloc = memref.AllocOp(t, [], []).result
                            value_map[op.result] = new_alloc
                            allocs_to_dealloc.append(new_alloc)
                            continue

                        if opname == "memref.dealloc":
                            # We'll emit deallocs at the end.
                            continue

                        # Default: clone operation and remap operands.
                        new_op = op.operation.clone()
                        for idx in range(len(new_op.operands)):
                            v = new_op.operands[idx]
                            if v in value_map:
                                new_op.operands[idx] = value_map[v]
                        for old_res, new_res in zip(op.results, new_op.results):
                            value_map[old_res] = new_res

                    # 3) DMA outputs back to external memory.
                    for i, barg in enumerate(core_func.entry_block.arguments):
                        if i not in output_arg_numbers:
                            continue
                        mt = MemRefType(barg.type)
                        rank = mt.rank
                        offsets = [c0] * rank
                        sizes = [arith.ConstantOp.create_index(int(d)) for d in mt.shape]
                        strides = [c1s] * rank

                        air_d.DmaMemcpyNd(
                            dst=herd_operands[i],
                            src=value_map[barg],
                            dst_offsets=offsets,
                            dst_sizes=sizes,
                            dst_strides=strides,
                            src_offsets=[],
                            src_sizes=[],
                            src_strides=[],
                        )

                    # 4) Dealloc L1 buffers.
                    for v in allocs_to_dealloc:
                        memref.DeallocOp(v)

                    # 5) Terminator.
                    air_d.HerdTerminatorOp()

                air_func.ReturnOp([])

        (project_dir / "top.mlir").write_text(str(dst_module), encoding="utf-8")
