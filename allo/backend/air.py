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

Additionally, this file contains a small Allo-MLIR -> AIR-MLIR translator
(:func:`convert`) used by `tests/dataflow/air/test_vector_from_allo.py`.

The translator is intentionally minimal: it targets the unit-test pattern
(Allo dataflow "single kernel" vector add) and generates an AIR `air.herd`
wrapping the compute region, with explicit DMA copies into L1 (mem space 2).

NOTE: This is MLIR-to-MLIR translation using the python MLIR bindings, not
string-based construction.
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


# ---------------------------------------------------------------------------
# Allo MLIR -> AIR MLIR conversion
# ---------------------------------------------------------------------------

from .._mlir.ir import Context as allo_Context, Module as allo_Module
from .._mlir.dialects import allo as allo_d


def _allo_type_to_air_type(allo_ty):
    """Convert a subset of Allo MLIR types into air.ir types."""

    from air.ir import (
        MemRefType as air_MemRefType,
        IntegerType as air_IntegerType,
        F16Type as air_F16Type,
        F32Type as air_F32Type,
        F64Type as air_F64Type,
        BF16Type as air_BF16Type,
    )

    if allo_ty.__class__.__name__ == "MemRefType":
        shape = list(allo_ty.shape)
        elem_str = str(allo_ty.element_type)

        if elem_str == "f16":
            air_elem = air_F16Type.get()
        elif elem_str == "f32":
            air_elem = air_F32Type.get()
        elif elem_str == "f64":
            air_elem = air_F64Type.get()
        elif elem_str == "bf16":
            air_elem = air_BF16Type.get()
        elif elem_str.startswith("ui"):
            air_elem = air_IntegerType.get_unsigned(int(elem_str[2:]))
        elif elem_str.startswith("si"):
            air_elem = air_IntegerType.get_signed(int(elem_str[2:]))
        elif elem_str.startswith("i"):
            air_elem = air_IntegerType.get_signless(int(elem_str[1:]))
        else:
            raise NotImplementedError(f"Unsupported element type: {elem_str}")

        # Allo memref types rarely carry non-default memory spaces in frontend output.
        mem_space_attr = None
        try:
            mem_space_attr = allo_ty.memory_space
        except Exception:
            mem_space_attr = None

        return air_MemRefType.get(shape, air_elem, memory_space=mem_space_attr)

    raise NotImplementedError(f"Unsupported type: {allo_ty} ({type(allo_ty)})")


def _convert_allo_attr_to_air_attr(attr):
    """Best-effort conversion of Allo IR attribute objects to AIR IR attributes."""

    from air.ir import (
        Attribute as air_Attribute,
        StringAttr as air_StringAttr,
        IntegerAttr as air_IntegerAttr,
        IntegerType as air_IntegerType,
        FloatAttr as air_FloatAttr,
        F32Type as air_F32Type,
        F64Type as air_F64Type,
    )

    if isinstance(attr, air_Attribute):
        return attr

    # Common attributes used in tests.
    cls = attr.__class__.__name__
    if cls == "StringAttr":
        return air_StringAttr.get(attr.value)

    if cls == "IntegerAttr":
        # `attr.type` is an allo IntegerType; use width and signedness via string.
        ty_str = str(attr.type)
        # default to signless.
        if ty_str.startswith("ui"):
            ty = air_IntegerType.get_unsigned(int(ty_str[2:]))
        elif ty_str.startswith("si"):
            ty = air_IntegerType.get_signed(int(ty_str[2:]))
        elif ty_str.startswith("i"):
            ty = air_IntegerType.get_signless(int(ty_str[1:]))
        else:
            ty = air_IntegerType.get_signless(32)
        return air_IntegerAttr.get(ty, attr.value)

    if cls == "FloatAttr":
        ty_str = str(attr.type)
        if ty_str == "f64":
            return air_FloatAttr.get(air_F64Type.get(), attr.value)
        return air_FloatAttr.get(air_F32Type.get(), attr.value)

    # Fallback: try to parse the textual form of the attribute in the AIR context.
    # This is not used for operation construction (only for copying attributes).
    try:
        from air.ir import Attribute as air_Attr

        return air_Attr.parse(str(attr))
    except Exception:
        raise NotImplementedError(f"Cannot convert attribute {attr} ({type(attr)})")


def convert(project: str):
    """Convert `<project>/allo.mlir` (Allo dialect) into `<project>/top.mlir` (AIR dialect).

    Supports the unit-test pattern:
    - a single `df.kernel` function containing `linalg.add` and `memref.copy`.

    Generated AIR IR is close to `tests/dataflow/air/vadd_prj/top.mlir`.
    """

    project_dir = _resolve_project_dir(project)
    allo_mlir_path = project_dir / "allo.mlir"
    if not allo_mlir_path.exists():
        raise FileNotFoundError(f"'{allo_mlir_path}' does not exist")

    content = allo_mlir_path.read_text(encoding="utf-8")

    # Parse Allo MLIR.
    with allo_Context() as ctx:
        allo_d.register_dialect(ctx)
        allo_mod = allo_Module.parse(str(content), ctx)

    kernel_func = None
    top_func = None
    for op in allo_mod.body.operations:
        # In Allo python bindings, `op.name` is the symbol name, while
        # `op.operation.name` is the operation name (e.g. "func.func").
        if op.operation.name != "func.func":
            continue
        attrs = op.attributes
        if "df.kernel" in attrs:
            kernel_func = op
        if "dataflow" in attrs:
            top_func = op

    if kernel_func is None:
        raise RuntimeError("convert(): no df.kernel function found in allo.mlir")
    if top_func is None:
        top_func = kernel_func

    # Identify linalg.add in kernel.
    linalg_add = None
    for bop in kernel_func.entry_block.operations:
        if bop.name == "linalg.add":
            linalg_add = bop
            break
    if linalg_add is None:
        raise RuntimeError("convert(): expected a linalg.add op in df.kernel")

    entry_name = "vector_add" if len(top_func.arguments) == 3 else "top"

    # Build AIR module.
    from air.ir import (
        InsertionPoint,
        IntegerType as air_IntegerType,
        IntegerAttr as air_IntegerAttr,
        FunctionType as air_FunctionType,
        MemRefType as air_MemRefType,
    )
    from air.dialects import func as func_d
    from air.dialects import arith as arith_d
    from air.dialects import memref as memref_d
    from air.dialects import linalg as linalg_d
    from air.dialects import air as air_d

    with Context() as _air_ctx, Location.unknown():
        air_module = Module.create()

        air_arg_types = [_allo_type_to_air_type(a.type) for a in top_func.arguments]
        ftype = air_FunctionType.get(air_arg_types, [])

        with InsertionPoint(air_module.body):
            f = func_d.FuncOp(entry_name, ftype)
            entry_block = f.add_entry_block()

            # Emit herd and return.
            with InsertionPoint(entry_block):
                herd = air_d.Herd(name="herd_0", sizes=[1, 1], operands=list(entry_block.arguments))

            herd_block = herd.body.blocks[0]
            # In herd block args: first 4 are tile ids and size args, then operands.
            herd_operands = list(herd_block.arguments)[4:]

            with InsertionPoint(herd_block):
                # Allocate local buffers in mem space 2 (L1).
                i32 = air_IntegerType.get_signless(32)
                ms2 = air_IntegerAttr.get(i32, 2)

                def _local_memref_type(global_mref_t: air_MemRefType) -> air_MemRefType:
                    return air_MemRefType.get(
                        list(global_mref_t.shape),
                        global_mref_t.element_type,
                        memory_space=ms2,
                    )

                locals_: list = []
                for arg in herd_operands:
                    gty = air_MemRefType(arg.type)
                    lty = _local_memref_type(gty)
                    loc_alloc = memref_d.AllocOp(lty, [], [])
                    locals_.append(loc_alloc.result)

                # DMA input(s) -> local. Assume last argument is output.
                out_idx = len(herd_operands) - 1

                # Only support 1D memrefs for this unit test.
                c0 = arith_d.ConstantOp.create_index(0)
                n = int(air_MemRefType(herd_operands[0].type).shape[0])
                cN = arith_d.ConstantOp.create_index(n)
                c1 = arith_d.ConstantOp.create_index(1)

                for i in range(min(2, out_idx)):
                    air_d.dma_memcpy_nd(
                        dst=locals_[i],
                        src=herd_operands[i],
                        dst_offsets=[],
                        dst_sizes=[],
                        dst_strides=[],
                        src_offsets=[c0.result],
                        src_sizes=[cN.result],
                        src_strides=[c1.result],
                    )

                # Compute: linalg.add local0 + local1 -> local_out.
                # For memref semantics, linalg.add has no tensor results.
                addop = linalg_d.AddOp([], inputs=[locals_[0], locals_[1]], outputs=[locals_[out_idx]])
                # Copy attributes (e.g. op_name = "add_0").
                for k, v in linalg_add.attributes.items():
                    addop.operation.attributes[k] = _convert_allo_attr_to_air_attr(v)

                # DMA local_out -> output memref.
                air_d.dma_memcpy_nd(
                    dst=herd_operands[out_idx],
                    src=locals_[out_idx],
                    dst_offsets=[c0.result],
                    dst_sizes=[cN.result],
                    dst_strides=[c1.result],
                    src_offsets=[],
                    src_sizes=[],
                    src_strides=[],
                )

                for v in locals_:
                    memref_d.DeallocOp(v)

                air_d.HerdTerminatorOp()

            with InsertionPoint(entry_block):
                func_d.ReturnOp([])

        (project_dir / "top.mlir").write_text(str(air_module), encoding="utf-8")
