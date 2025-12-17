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
single-kernel vector addition style programs used in unit tests.
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


def convert(project: str):
    """Convert `<project>/allo.mlir` (Allo MLIR) into `<project>/top.mlir` (AIR MLIR).

    This is an MLIR-level dialect-to-dialect lowering used by unit tests.

    Currently supported pattern (sufficient for `test_vector_from_allo.py`):
      - one `df.kernel` function containing
            %tmp = memref.alloc
            linalg.add ins(%arg0, %arg1) outs(%tmp)
            memref.copy %tmp, %arg2
      - one `dataflow` top function with the desired argument list.

    The conversion produces a `func.func @top` that wraps the kernel body in a
    single-tile `air.herd`, inserts explicit DMA copies between external memory
    and herd-local memories (memref memory space 2), and deallocates the local
    buffers.

    Implementation notes:
      - Avoids textual MLIR construction.
      - Re-creates required linalg generic region so that the produced MLIR
        parses correctly.
    """

    project_dir = _resolve_project_dir(project)
    allo_mlir_path = project_dir / "allo.mlir"
    if not allo_mlir_path.exists():
        raise FileNotFoundError(f"'{allo_mlir_path}' does not exist")

    src_txt = allo_mlir_path.read_text(encoding="utf-8")

    # Use mlir-air's python bindings to parse and construct IR in one context.
    from air.ir import (
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        MemRefType,
        F16Type,
        F32Type,
        F64Type,
        BF16Type,
    )
    from air.dialects import func, arith, memref, linalg, air as air_d

    def _is_float_type(t) -> bool:
        return isinstance(t, (F16Type, F32Type, F64Type, BF16Type))

    with Context() as ctx, Location.unknown():
        src_mod = Module.parse(src_txt, ctx)

        # Find the df.kernel function (e.g., core_0) and the dataflow top.
        kernel_func = None
        top_sig_func = None
        for op in src_mod.body.operations:
            if op.operation.name != "func.func":
                continue
            attrs = op.attributes
            if "df.kernel" in attrs:
                kernel_func = op
            if "dataflow" in attrs and attrs["sym_name"].value == "top":
                top_sig_func = op

        if kernel_func is None:
            raise ValueError("No 'df.kernel' function found in allo.mlir")
        if top_sig_func is None:
            raise ValueError("No 'func.func @top' with 'dataflow' attribute found in allo.mlir")

        # Extract kernel computation info.
        kblock = kernel_func.regions[0].blocks[0]
        linalg_add_op = None
        memref_copy_op = None
        for bop in kblock.operations:
            if bop.operation.name == "linalg.add":
                linalg_add_op = bop
            elif bop.operation.name == "memref.copy":
                memref_copy_op = bop
        if linalg_add_op is None or memref_copy_op is None:
            raise ValueError(
                "Unsupported kernel pattern: expected 'linalg.add' and 'memref.copy' in kernel body"
            )

        # Copy linalg.add attributes we care about (e.g., op_name).
        add_attrs = dict(linalg_add_op.attributes)

        # Build AIR module.
        air_mod = Module.create()
        with InsertionPoint(air_mod.body):
            # Create top function with the same signature as the Allo top.
            ftype = top_sig_func.attributes["function_type"].value
            top = func.FuncOp("top", ftype)
            entry = top.add_entry_block()

            with InsertionPoint(entry):
                # Herd sizes: 1x1 for unit tests.
                c1 = arith.ConstantOp.create_index(1)
                c1_0 = arith.ConstantOp.create_index(1)

                herd = air_d.Herd(
                    name="herd_0",
                    sizes=[c1.result, c1_0.result],
                    operands=list(entry.arguments),
                )

                # Herd body block args: [tile_ids(2), sizes(2), operands...]
                hblock = herd.body.blocks[0]

                # External memrefs passed into herd.
                if len(entry.arguments) != 3:
                    raise ValueError(
                        f"Unsupported top signature: expected 3 memref arguments, got {len(entry.arguments)}"
                    )
                if len(hblock.arguments) < 4 + len(entry.arguments):
                    raise ValueError("Unexpected herd block signature")

                argA = hblock.arguments[4]
                argB = hblock.arguments[5]
                argC = hblock.arguments[6]

                # Create herd-local memrefs in memory space 2 (AIE local).
                i32 = IntegerType.get_signless(32)
                ms2 = IntegerAttr.get(i32, 2)

                def with_memspace2(mref_ty):
                    mty = MemRefType(mref_ty)
                    return MemRefType.get(
                        mty.shape,
                        mty.element_type,
                        layout=mty.layout,
                        memory_space=ms2,
                    )

                localA_ty = with_memspace2(argA.type)
                localB_ty = with_memspace2(argB.type)
                localC_ty = with_memspace2(argC.type)

                with InsertionPoint(hblock):
                    localA = memref.AllocOp(localA_ty, [], []).result
                    localB = memref.AllocOp(localB_ty, [], []).result
                    localC = memref.AllocOp(localC_ty, [], []).result

                    # DMA in (global -> local)
                    c0 = arith.ConstantOp.create_index(0)

                    # Assume 1D memref for the unit test.
                    shape = MemRefType(argA.type).shape
                    if len(shape) != 1:
                        raise ValueError(
                            f"Unsupported memref rank {len(shape)}: only rank-1 is supported by this converter"
                        )
                    n = shape[0]
                    cN = arith.ConstantOp.create_index(n)
                    c1_s = arith.ConstantOp.create_index(1)

                    air_d.dma_memcpy_nd(
                        localA,
                        argA,
                        src_offsets=[c0.result],
                        src_sizes=[cN.result],
                        src_strides=[c1_s.result],
                    )
                    air_d.dma_memcpy_nd(
                        localB,
                        argB,
                        src_offsets=[c0.result],
                        src_sizes=[cN.result],
                        src_strides=[c1_s.result],
                    )

                    # Compute (linalg.add)
                    add_op = linalg.AddOp([], [localA, localB], [localC])

                    # Ensure the op has a valid region (generic form) so it round-trips.
                    if len(add_op.regions[0].blocks) == 0:
                        et = MemRefType(localA.type).element_type
                        add_op.regions[0].blocks.append(et, et, et)
                        blk = add_op.regions[0].blocks[0]
                        with InsertionPoint(blk):
                            if _is_float_type(et):
                                r = arith.AddFOp(blk.arguments[0], blk.arguments[1]).result
                            else:
                                r = arith.AddIOp(blk.arguments[0], blk.arguments[1]).result
                            linalg.YieldOp([r])

                    for k, v in add_attrs.items():
                        add_op.operation.attributes[k] = v

                    # DMA out (local -> global)
                    air_d.dma_memcpy_nd(
                        argC,
                        localC,
                        dst_offsets=[c0.result],
                        dst_sizes=[cN.result],
                        dst_strides=[c1_s.result],
                    )

                    memref.DeallocOp(localA)
                    memref.DeallocOp(localB)
                    memref.DeallocOp(localC)
                    air_d.HerdTerminatorOp()

                func.ReturnOp([])

        # Write output.
        top_mlir_path = project_dir / "top.mlir"
        top_mlir_path.write_text(str(air_mod), encoding="utf-8")
