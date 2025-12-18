# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend.

This backend performs **MLIR-to-MLIR** lowering from the MLIR produced by the
Allo frontend to an MLIR-AIR module suitable for `aircc`.

The unit tests in `tests/dataflow/air/` expect:

* `df.build(..., target="air")` produces a project directory containing:
  - `allo.mlir`: the incoming Allo MLIR
  - `top.mlir`: the translated AIR MLIR
* The returned module object is callable and executes the generated AIR project
  using XRT via :func:`_call_prj`.

The translation implemented here is intentionally minimal and targets the vector
kernels used by the unit tests:

* Single `df.kernel` function in the module.
* Memref + linalg ops inside the kernel.
* No `allo.*` operations in the kernel body.

Implementation notes
--------------------
We avoid string matching for *operation construction*.

* Kernel compute ops are cloned into the AIR herd body using `op.clone()`.
* Buffer allocations are re-created (not cloned) to change memory space to 2.
* `memref.copy` *into an output argument* is not cloned; instead we translate it
  into the final output DMA source (matching the reference AIR IR in tests).

"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path
from typing import Any

import filelock
import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import Context as AirContext
from air.ir import Location as AirLocation
from air.ir import Module as AirModuleIR

# Register dialects by importing them.
from air.dialects import air as air_d  # noqa: F401
from air.dialects import arith as air_arith
from air.dialects import func as air_func
from air.dialects import linalg as air_linalg  # noqa: F401
from air.dialects import memref as air_memref
from air.ir import Block as AirBlock
from air.ir import InsertionPoint
from air.ir import IndexType
from air.ir import IntegerAttr
from air.ir import IntegerType

# aircc driver (compiler)
import air.compiler.aircc.main as aircc

# Allo-side helpers (for analysis)
from .._mlir.ir import Context as AlloContext
from .._mlir.ir import Location as AlloLocation
from .._mlir.ir import Module as AlloModuleIR
from .._mlir.dialects import allo as allo_d
from .._mlir.dialects import func as allo_func_d
from ..passes import analyze_read_write_patterns


# -----------------------------------------------------------------------------
# Project directory helpers
# -----------------------------------------------------------------------------


def _resolve_project_dir(project: str) -> Path:
    """Resolve a *pre-existing* project directory from a potentially relative name."""

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

    raise FileNotFoundError(
        f"AIR project directory '{project}' not found. Tried CWD='{Path.cwd()}', caller-relative, and repo-relative paths."
    )


def _resolve_project_dir_for_write(project: str) -> Path:
    """Resolve a project directory for writing.

    The runtime sandbox blocks *relative* file IO from library code. For `df.build`
    we therefore:

    * always return an **absolute** path
    * prefer a path relative to the *user/test* call site, not inside the `allo`
      package implementation.
    """

    p = Path(project)
    if p.is_absolute():
        return p

    pkg_root = Path(__file__).resolve().parents[1]  # .../allo/allo

    # Find the first stack frame that is *outside* the Allo package.
    for frame_info in inspect.stack()[1:]:
        f = frame_info.filename
        if not f:
            continue
        fp = Path(f).resolve()
        if fp.is_relative_to(pkg_root):
            continue
        return (fp.parent / p).resolve()

    # Fallback: CWD
    return (Path.cwd() / p).resolve()


# -----------------------------------------------------------------------------
# Project runner (used by tests)
# -----------------------------------------------------------------------------


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile and execute an AIR project."""

    def _compile_external_kernel(src: Path, project_dir: Path) -> Path:
        clangpp = Path(os.environ.get("PEANO_INSTALL_DIR")) / "bin" / "clang++"
        if not clangpp.exists():
            raise RuntimeError(f"Cannot find '{clangpp}'.")

        mlir_aie = os.environ.get("MLIR_AIE_INSTALL_DIR")
        if not mlir_aie:
            raise RuntimeError(
                "MLIR_AIE_INSTALL_DIR is not set (required for external kernel includes)."
            )
        target = "aie2p" if os.getenv("NPU2") == "1" else "aie2"
        cmd: list[str] = [
            str(clangpp),
            "-O2",
            "-std=c++20",
            f"--target={target}-none-unknown-elf",
            "-Wno-parentheses",
            "-Wno-attributes",
            "-Wno-macro-redefined",
            "-DNDEBUG",
            "-I",
            str(Path(mlir_aie) / "include"),
            "-I",
            str(project_dir),
            "-I",
            os.environ.get("MLIR_AIE_EXTERNAL_KERNEL_DIR"),
            "-c",
            str(src),
            "-o",
            str(project_dir / f"{src.name}.o"),
        ]

        proc = subprocess.run(cmd, cwd=str(project_dir), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to compile external kernel.\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )

    def _compile_air_project(
        air_module: AirModuleIR,
        project_dir: Path,
        build_dir: Path,
        xclbin_path: Path,
        insts_path: Path,
    ) -> XRTCompileArtifact:
        tmpdir = build_dir / "aircc_tmp"
        tmpdir.mkdir(parents=True, exist_ok=True)

        aircc_options: list[str] = [
            "--device",
            "npu2" if os.getenv("NPU2") == "1" else "npu1",
            "air.mlir",
            "--tmpdir",
            str(tmpdir),
            "-o",
            str(xclbin_path),
            "-i",
            str(insts_path),
            "--output-format",
            "xclbin",
            "--peano",
            os.environ.get("PEANO_INSTALL_DIR"),
            "--no-xchesscc",
            "--no-xbridge",
        ]

        old_cwd = os.getcwd()
        try:
            os.chdir(str(project_dir))
            aircc.run(air_module, aircc_options)
        finally:
            os.chdir(old_cwd)

        return XRTCompileArtifact(str(xclbin_path), "MLIR_AIE", str(insts_path))

    project_dir = _resolve_project_dir(project)
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

    with filelock.FileLock(str(prj_lock_path)):
        srcs = sorted(project_dir.glob("*.cc"))
        srcs = [p for p in srcs if p.is_file() and p.parent == project_dir]
        for src in srcs:
            _compile_external_kernel(src, project_dir)

        mlir_txt = top_mlir.read_text(encoding="utf-8")
        with AirContext() as ctx, AirLocation.unknown():
            air_module = AirModuleIR.parse(mlir_txt, ctx)

        artifact = _compile_air_project(
            air_module,
            project_dir=project_dir,
            build_dir=build_dir,
            xclbin_path=build_dir / "final.xclbin",
            insts_path=build_dir / "insts.bin",
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
# MLIR-to-MLIR conversion: Allo (std dialect subset) -> AIR
# -----------------------------------------------------------------------------


def _find_allo_kernel_io(
    allo_module: AlloModuleIR, top_func_name: str
) -> tuple[list[int], list[int]]:
    top_kernel = None
    for op in allo_module.body.operations:
        if isinstance(op, allo_func_d.FuncOp) and "df.kernel" in op.attributes:
            top_kernel = op
            break
    if top_kernel is None:
        raise RuntimeError("AIR backend: no function with 'df.kernel' attribute found")

    in_idx, out_idx = analyze_read_write_patterns(top_kernel)
    out_set = set(out_idx)
    in_set = set(in_idx)
    pure_in = sorted(list(in_set - out_set))
    pure_out = sorted(list(out_set))
    return pure_in, pure_out


def convert(
    module: Any, top_func_name: str, project_dir: Path
) -> tuple[str, list[int]]:
    project_dir.mkdir(parents=True, exist_ok=True)

    with AlloContext() as ctx, AlloLocation.unknown():
        allo_d.register_dialect(ctx)
        allo_mod = AlloModuleIR.parse(str(module), ctx)
        _, output_idx = _find_allo_kernel_io(allo_mod, top_func_name)

    with AirContext() as ctx, AirLocation.unknown():
        src = AirModuleIR.parse(str(module), ctx)

        def _find_func(sym_name: str):
            for op in src.body.operations:
                if op.operation.name != "func.func":
                    continue
                if "sym_name" not in op.attributes:
                    continue
                if op.attributes["sym_name"].value == sym_name:
                    return op
            return None

        src_top = _find_func(top_func_name)
        if src_top is None:
            raise RuntimeError(f"AIR backend: cannot find top func '{top_func_name}'")

        src_kernel = None
        for op in src.body.operations:
            if op.operation.name != "func.func":
                continue
            if "df.kernel" in op.attributes:
                src_kernel = op
                break
        if src_kernel is None:
            raise RuntimeError("AIR backend: cannot find df.kernel func")

        dst = AirModuleIR.create()

        idx_ty = IndexType.get()
        const_cache: dict[int, Any] = {}

        def cidx(v: int):
            if v not in const_cache:
                const_cache[v] = air_arith.ConstantOp(idx_ty, v)
            return const_cache[v].result

        i32_ty = IntegerType.get_signless(32)
        ms2 = IntegerAttr.get(i32_ty, 2)

        def memref_with_memspace2(memref_ty):
            return air_memref.MemRefType.get(
                memref_ty.shape,
                memref_ty.element_type,
                memref_ty.layout,
                ms2,
            )

        with InsertionPoint(dst.body):
            topf = air_func.FuncOp(top_func_name, src_top.type)
            entry = topf.add_entry_block()

        with InsertionPoint(entry):
            one0 = air_arith.ConstantOp(idx_ty, 1)
            one1 = air_arith.ConstantOp(idx_ty, 1)
            herd = air_d.HerdOp(
                async_token=None,
                async_dependencies=[],
                sizes=[one0.result, one1.result],
                herd_operands=list(entry.arguments),
                sym_name="herd_0",
            )

            # Herd body args are: (tile_x, tile_y, size_x, size_y, *operands)
            herd_arg_types = [idx_ty, idx_ty, idx_ty, idx_ty] + [
                a.type for a in entry.arguments
            ]
            hb = AirBlock.create_at_start(herd.body, herd_arg_types)
            air_func.ReturnOp([])

        with InsertionPoint(hb):
            herd_args = list(hb.arguments[4:])
            vmap: dict[Any, Any] = {}
            local_kernel_args: list[Any] = []

            for i, ha in enumerate(herd_args):
                if ha.type.__class__.__name__ != "MemRefType":
                    local_kernel_args.append(ha)
                    vmap[src_kernel.arguments[i]] = ha
                    continue
                loc_ty = memref_with_memspace2(ha.type)
                alloc = air_memref.AllocOp(loc_ty, [], [])
                local_kernel_args.append(alloc.result)
                vmap[src_kernel.arguments[i]] = alloc.result

            pure_out = set(output_idx)
            final_out_src: dict[int, Any] = {
                i: local_kernel_args[i] for i in pure_out if i < len(local_kernel_args)
            }

            # Input DMAs
            for i, ha in enumerate(herd_args):
                if i in pure_out:
                    continue
                if ha.type.__class__.__name__ != "MemRefType":
                    continue
                shape = ha.type.shape
                rank = len(shape)
                air_d.DmaMemcpyNdOp(
                    async_token=None,
                    async_dependencies=[],
                    dst=local_kernel_args[i],
                    dst_offsets=[],
                    dst_sizes=[],
                    dst_strides=[],
                    src=ha,
                    src_offsets=[cidx(0) for _ in range(rank)],
                    src_sizes=[cidx(int(d)) for d in shape],
                    src_strides=[cidx(1) for _ in range(rank)],
                )

            allocs_to_dealloc: list[Any] = [
                a
                for a in local_kernel_args
                if a.type.__class__.__name__ == "MemRefType"
            ]

            kernel_arg_to_idx = {
                src_kernel.arguments[i]: i for i in range(len(src_kernel.arguments))
            }

            for op in src_kernel.regions[0].blocks[0].operations:
                if op.operation.name == "func.return":
                    break

                if op.operation.name == "memref.alloc":
                    old_res = op.results[0]
                    new_ty = memref_with_memspace2(old_res.type)
                    new_alloc = air_memref.AllocOp(new_ty, [], [])
                    for k, v in op.attributes.items():
                        new_alloc.operation.attributes[k] = v
                    vmap[old_res] = new_alloc.result
                    allocs_to_dealloc.append(new_alloc.result)
                    continue

                if op.operation.name == "memref.copy":
                    src_val = vmap.get(op.operands[0], op.operands[0])
                    dst_val = op.operands[1]
                    if dst_val in kernel_arg_to_idx:
                        dst_idx = kernel_arg_to_idx[dst_val]
                        if dst_idx in pure_out:
                            final_out_src[dst_idx] = src_val
                            continue

                cloned = op.clone()
                for idx, operand in enumerate(op.operands):
                    if operand in vmap:
                        cloned.operands[idx] = vmap[operand]
                for old_r, new_r in zip(op.results, cloned.results):
                    vmap[old_r] = new_r

            # Output DMAs
            for i in sorted(pure_out):
                ha = herd_args[i]
                if ha.type.__class__.__name__ != "MemRefType":
                    continue
                shape = ha.type.shape
                rank = len(shape)
                air_d.DmaMemcpyNdOp(
                    async_token=None,
                    async_dependencies=[],
                    dst=ha,
                    dst_offsets=[cidx(0) for _ in range(rank)],
                    dst_sizes=[cidx(int(d)) for d in shape],
                    dst_strides=[cidx(1) for _ in range(rank)],
                    src=final_out_src[i],
                    src_offsets=[],
                    src_sizes=[],
                    src_strides=[],
                )

            for buf in reversed(allocs_to_dealloc):
                air_memref.DeallocOp(buf)

            air_d.HerdTerminatorOp()

        air_txt = str(dst)

    (project_dir / "top.mlir").write_text(air_txt, encoding="utf-8")
    return air_txt, output_idx


# -----------------------------------------------------------------------------
# Public module wrapper
# -----------------------------------------------------------------------------


class AIRModule:
    """A minimal wrapper around an AIR project directory."""

    def __init__(
        self,
        module,
        top_func_name: str,
        ext_libs=None,
        mode=None,
        project=None,
        configs=None,
        func_args=None,
        wrap_io=True,
    ):
        self.top_func_name = top_func_name
        self.ext_libs = ext_libs or []
        self.mode = mode
        self.configs = configs or {}
        self.func_args = func_args
        self.wrap_io = wrap_io

        self.project = project or f"{top_func_name}.prj"
        self.project_dir = _resolve_project_dir_for_write(self.project)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        (self.project_dir / "allo.mlir").write_text(str(module), encoding="utf-8")
        _, self.output_idx = convert(module, top_func_name, self.project_dir)

    def __repr__(self):
        return f"AIRModule(top={self.top_func_name}, project={self.project_dir})"

    def get_ir(self) -> str:
        return (self.project_dir / "top.mlir").read_text(encoding="utf-8")

    def __call__(self, *args):
        return _call_prj(str(self.project_dir), self.output_idx, *args)


def build(
    module,
    top_func_name,
    ext_libs,
    mode,
    project,
    configs,
    func_args,
    wrap_io,
):
    """Entry point used by :meth:`allo.customize.Schedule.build`."""

    return AIRModule(
        module=module,
        top_func_name=top_func_name,
        ext_libs=ext_libs,
        mode=mode,
        project=project,
        configs=configs,
        func_args=func_args,
        wrap_io=wrap_io,
    )
