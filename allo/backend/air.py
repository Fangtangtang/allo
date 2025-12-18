# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend.

This backend implements a small MLIR-to-MLIR translation from Allo's MLIR to
mlir-air MLIR, sufficient for the dataflow AIR vector tests.

Design notes
------------
* The dataflow frontend may temporarily produce IR that does not verify
  (e.g., `top` calling per-tile kernels with operand type mismatches). This
  backend therefore **does not** re-parse the input module from text.
* IR construction uses Python op constructors (no string-based construction).
* The generated AIR module uses:
  - `air.herd` for a 1xP array of tiles (P = number of df.kernel instances)
  - `air.dma_memcpy_nd` for 1D host<->tile transfers
  - `memref`/`linalg`/`arith` for compute

Supported kernel ops (current tests)
------------------------------------
- arith.constant
- memref.alloc / memref.dealloc
- memref.copy
- linalg.add / linalg.mul
- linalg.fill / linalg.broadcast

The linalg named ops in AIR expect a region; we build minimal regions matching
Allo's elementwise semantics.
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
from air.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    IntegerType,
    IntegerAttr,
    IndexType,
    MemRefType,
    F16Type,
    F32Type,
    F64Type,
    Type,
)

from air.dialects import air as air_d
from air.dialects import func as func_d
from air.dialects import arith as arith_d
from air.dialects import memref as memref_d
from air.dialects import linalg as linalg_d

import air.compiler.aircc.main as aircc

# Allo MLIR bindings (source dialect/module)
import allo._mlir._mlir_libs._mlir as allo_ir
from .._mlir.dialects import func as allo_func_d
from .._mlir.dialects import memref as allo_memref_d
from .._mlir.dialects import linalg as allo_linalg_d
from .._mlir.dialects import arith as allo_arith_d
from .._mlir.ir import (
    MemRefType as AlloMemRefType,
    IntegerType as AlloIntegerType,
    IndexType as AlloIndexType,
    IntegerAttr as AlloIntegerAttr,
    FloatAttr as AlloFloatAttr,
    F16Type as AlloF16Type,
    F32Type as AlloF32Type,
    F64Type as AlloF64Type,
    BlockArgument,
)


# -----------------------------------------------------------------------------
# Path helpers (write restrictions)
# -----------------------------------------------------------------------------

_ALLOWED_WRITE_ROOT = Path("/home/sf668/workspace/allo/tests").resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except Exception:
        return False


def _iter_callers() -> list[Path]:
    callers: list[Path] = []
    for frame_info in inspect.stack()[2:]:
        f = frame_info.filename
        if not f:
            continue
        fp = Path(f).resolve()
        if fp == Path(__file__).resolve():
            continue
        callers.append(fp.parent)

    callers.sort(key=lambda p: (0 if _is_under(p, _ALLOWED_WRITE_ROOT) else 1, len(str(p))))
    return callers


def _resolve_project_dir(project: str) -> Path:
    """Resolve an existing project directory."""

    p = Path(project)
    if p.is_absolute():
        return p

    cands: list[Path] = []
    for base in _iter_callers():
        cand = (base / p).resolve()
        if cand.exists():
            cands.append(cand)

    if p.exists():
        cands.append(p.resolve())

    if not cands:
        raise FileNotFoundError(
            f"AIR project directory '{project}' not found. Tried caller-relative paths and CWD='{Path.cwd()}'."
        )

    for cand in cands:
        if _is_under(cand, _ALLOWED_WRITE_ROOT):
            return cand
    return cands[0]


def _resolve_or_create_project_dir(project: str) -> Path:
    """Resolve a project directory and ensure it is writable."""

    p = Path(project)
    if p.is_absolute():
        if not _is_under(p, _ALLOWED_WRITE_ROOT):
            raise PermissionError(
                f"AIR backend cannot write to '{p}'. Allowed root: '{_ALLOWED_WRITE_ROOT}'."
            )
        p.mkdir(parents=True, exist_ok=True)
        return p

    try:
        resolved = _resolve_project_dir(project)
        if _is_under(resolved, _ALLOWED_WRITE_ROOT):
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
    except FileNotFoundError:
        pass

    base = _ALLOWED_WRITE_ROOT / "_air_prj"
    base.mkdir(parents=True, exist_ok=True)
    resolved = (base / project).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# -----------------------------------------------------------------------------
# Runner (used by tests)
# -----------------------------------------------------------------------------


def _call_prj(project: str, output_idx: list[int], *args):
    """Compile and execute an AIR project (top.mlir) under `project`."""

    def _compile_external_kernel(src: Path, project_dir: Path) -> None:
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
        air_module: Module,
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
        for src in sorted(project_dir.glob("*.cc")):
            if src.is_file() and src.parent == project_dir:
                _compile_external_kernel(src, project_dir)

        mlir_txt = top_mlir.read_text(encoding="utf-8")
        with Context() as ctx, Location.unknown():
            air_module = Module.parse(mlir_txt, ctx)

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
# Helpers for AIR construction
# -----------------------------------------------------------------------------


def _dense_i64_array_to_list(attr: Any) -> list[int]:
    if attr is None:
        return []
    res: list[int] = []
    try:
        it = list(attr)
    except Exception:
        return []
    for x in it:
        if isinstance(x, int):
            res.append(x)
        elif hasattr(x, "value"):
            res.append(int(x.value))
        else:
            res.append(int(str(x)))
    return res


def _int_attr_i32(value: int) -> IntegerAttr:
    return IntegerAttr.get(IntegerType.get_signless(32), value)


def _is_float_type(t: Type) -> bool:
    return isinstance(t, (F16Type, F32Type, F64Type))


def _build_linalg_add_region(op: linalg_d.AddOp, elem_ty: Type):
    if len(op.regions[0].blocks) == 0:
        op.regions[0].blocks.append(elem_ty, elem_ty, elem_ty)
    b = op.regions[0].blocks[0]
    with InsertionPoint(b):
        if _is_float_type(elem_ty):
            r = arith_d.AddFOp(b.arguments[0], b.arguments[1]).result
        else:
            r = arith_d.AddIOp(b.arguments[0], b.arguments[1]).result
        linalg_d.YieldOp([r])


def _build_linalg_mul_region(op: linalg_d.MulOp, elem_ty: Type):
    if len(op.regions[0].blocks) == 0:
        op.regions[0].blocks.append(elem_ty, elem_ty, elem_ty)
    b = op.regions[0].blocks[0]
    with InsertionPoint(b):
        if _is_float_type(elem_ty):
            r = arith_d.MulFOp(b.arguments[0], b.arguments[1]).result
        else:
            r = arith_d.MulIOp(b.arguments[0], b.arguments[1]).result
        linalg_d.YieldOp([r])


def _build_linalg_fill_region(op: linalg_d.FillOp, elem_ty: Type):
    if len(op.regions[0].blocks) == 0:
        op.regions[0].blocks.append(elem_ty, elem_ty)
    b = op.regions[0].blocks[0]
    with InsertionPoint(b):
        linalg_d.YieldOp([b.arguments[0]])


def _build_linalg_broadcast_region(op: linalg_d.BroadcastOp, elem_ty: Type):
    if len(op.regions[0].blocks) == 0:
        op.regions[0].blocks.append(elem_ty, elem_ty)
    b = op.regions[0].blocks[0]
    with InsertionPoint(b):
        linalg_d.YieldOp([b.arguments[0]])


def _convert_allo_type_to_air(ty: Any) -> Any:
    """Convert a type object from Allo Python bindings to air.ir types."""

    if isinstance(ty, AlloMemRefType):
        elem = _convert_allo_type_to_air(ty.element_type)
        ms = None
        if ty.memory_space is not None:
            if isinstance(ty.memory_space, AlloIntegerAttr):
                ms = _int_attr_i32(int(ty.memory_space.value))
            else:
                ms = _int_attr_i32(int(str(ty.memory_space)))
        return MemRefType.get(tuple(ty.shape), elem, None, ms)

    if isinstance(ty, AlloIndexType):
        return IndexType.get()

    if isinstance(ty, AlloIntegerType):
        w = int(ty.width)
        if getattr(ty, "is_unsigned", False):
            return IntegerType.get_unsigned(w)
        if getattr(ty, "is_signed", False):
            return IntegerType.get_signed(w)
        return IntegerType.get_signless(w)

    if isinstance(ty, AlloF16Type):
        return F16Type.get()
    if isinstance(ty, AlloF32Type):
        return F32Type.get()
    if isinstance(ty, AlloF64Type):
        return F64Type.get()

    raise TypeError(f"Unsupported type conversion from Allo to AIR: {ty} ({type(ty)})")


def _get_df_kernels(allo_module: allo_ir.ir.Module) -> list[allo_func_d.FuncOp]:
    kernels: list[allo_func_d.FuncOp] = []
    for op in allo_module.body.operations:
        if isinstance(op, allo_func_d.FuncOp) and "df.kernel" in op.attributes:
            kernels.append(op)
    return kernels


def _find_func(allo_module: allo_ir.ir.Module, name: str) -> allo_func_d.FuncOp | None:
    for op in allo_module.body.operations:
        if isinstance(op, allo_func_d.FuncOp) and op.attributes["sym_name"].value == name:
            return op
    return None


def _analyze_kernel_arg_rw(kernel: allo_func_d.FuncOp) -> tuple[set[int], set[int]]:
    """Return (read_arg_indices, write_arg_indices) for memref arguments."""

    reads: set[int] = set()
    writes: set[int] = set()

    def mark_arg(val, is_write: bool):
        if not BlockArgument.isinstance(val):
            return
        barg = BlockArgument(val)
        if not isinstance(barg.owner.owner, allo_func_d.FuncOp):
            return
        idx = barg.arg_number
        if not isinstance(kernel.arguments[idx].type, AlloMemRefType):
            return
        if is_write:
            writes.add(idx)
        else:
            reads.add(idx)

    for blk in kernel.body.blocks:
        for op in blk.operations:
            name = op.operation.name
            if name == "memref.copy":
                if len(op.operands) >= 2:
                    mark_arg(op.operands[0], is_write=False)
                    mark_arg(op.operands[1], is_write=True)
            elif name in {"linalg.add", "linalg.mul"}:
                if len(op.operands) >= 3:
                    mark_arg(op.operands[0], is_write=False)
                    mark_arg(op.operands[1], is_write=False)
                    mark_arg(op.operands[2], is_write=True)
            elif name == "linalg.fill":
                if len(op.operands) >= 2:
                    mark_arg(op.operands[0], is_write=False)
                    mark_arg(op.operands[1], is_write=True)
            elif name == "linalg.broadcast":
                if len(op.operands) >= 2:
                    mark_arg(op.operands[0], is_write=False)
                    mark_arg(op.operands[1], is_write=True)

    return reads, writes


class AIRModule:
    """A compiled AIR project wrapper."""

    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name: str,
        ext_libs=None,
        mode=None,
        project: str | None = None,
        configs=None,
        func_args=None,
        wrap_io: bool = True,
    ):
        self.top_func_name = top_func_name
        self.ext_libs = ext_libs or []
        self.mode = mode
        self.project = project or f"{top_func_name}.prj"
        self.configs = configs or {}
        self.func_args = func_args
        self.wrap_io = wrap_io

        # IMPORTANT: do not re-parse `module` from text; dataflow may produce
        # temporarily invalid IR (e.g., mismatched call operand types).
        self.allo_module: allo_ir.ir.Module = module

        self.project_dir = _resolve_or_create_project_dir(self.project)
        self.output_idx: list[int] = []
        self.air_module: Module | None = None
        self._build_air_project()

    def __repr__(self):
        return f"AIRModule(top={self.top_func_name}, project={self.project_dir})"

    def get_ir(self) -> str:
        return str(self.air_module) if self.air_module is not None else ""

    def _build_air_project(self) -> None:
        top_func = _find_func(self.allo_module, self.top_func_name)
        if top_func is None:
            raise RuntimeError(f"Top function '{self.top_func_name}' not found in Allo module")

        kernels = _get_df_kernels(self.allo_module)
        if not kernels:
            raise RuntimeError("No df.kernel functions found in Allo module")

        herd_y = len(kernels)
        kernel = kernels[0]
        reads, writes = _analyze_kernel_arg_rw(kernel)

        # output indices correspond to top function args (same order as kernel args in tests)
        self.output_idx = sorted(list(writes))

        tile_len = None
        for a in kernel.arguments:
            if isinstance(a.type, AlloMemRefType):
                shape = tuple(a.type.shape)
                if len(shape) != 1:
                    raise NotImplementedError("AIR backend currently supports 1D memrefs only")
                tile_len = int(shape[0])
                break
        if tile_len is None:
            raise RuntimeError("Cannot determine tile size from kernel signature")

        with Context() as _ctx, Location.unknown():
            self.air_module = Module.create()

            in_types = [_convert_allo_type_to_air(a.type) for a in top_func.arguments]
            func_type = func_d.FunctionType.get(in_types, [])

            with InsertionPoint(self.air_module.body):
                air_top = func_d.FuncOp(self.top_func_name, func_type)
                entry = air_top.add_entry_block()

                with InsertionPoint(entry):
                    idx_ty = IndexType.get()
                    c1 = arith_d.ConstantOp(idx_ty, 1).result
                    cHerdY = arith_d.ConstantOp(idx_ty, herd_y).result

                    @air_d.herd(name="herd_0", sizes=[c1, cHerdY], operands=list(entry.arguments))
                    def _herd_body(*hb_args):
                        tile_y = hb_args[1]
                        herd_operands = list(hb_args[4:])

                        cTile = arith_d.ConstantOp(idx_ty, tile_len).result
                        cStride1 = arith_d.ConstantOp(idx_ty, 1).result
                        base_offset = arith_d.MulIOp(tile_y, cTile).result

                        ms2 = _int_attr_i32(2)

                        vmap: dict[Any, Any] = {}
                        local_args: dict[int, Any] = {}

                        # Allocate local buffers for each kernel memref arg.
                        for i, (karg, garg) in enumerate(zip(kernel.arguments, herd_operands)):
                            if not isinstance(karg.type, AlloMemRefType):
                                continue

                            elem_ty = _convert_allo_type_to_air(karg.type.element_type)
                            local_ty = MemRefType.get(tuple(karg.type.shape), elem_ty, None, ms2)
                            loc_buf = memref_d.AllocOp(local_ty, [], []).result
                            local_args[i] = loc_buf
                            vmap[karg] = loc_buf

                            if i in reads:
                                air_d.dma_memcpy_nd(
                                    dst=loc_buf,
                                    src=garg,
                                    src_offsets=[base_offset],
                                    src_sizes=[cTile],
                                    src_strides=[cStride1],
                                )

                        # Compute.
                        self._lower_kernel_into_herd(kernel, vmap, ms2, set(local_args.values()))

                        # Writeback outputs.
                        for i, garg in enumerate(herd_operands):
                            if i not in writes:
                                continue
                            loc_buf = local_args.get(i)
                            if loc_buf is None:
                                continue
                            air_d.dma_memcpy_nd(
                                dst=garg,
                                src=loc_buf,
                                dst_offsets=[base_offset],
                                dst_sizes=[cTile],
                                dst_strides=[cStride1],
                            )

                        for buf in local_args.values():
                            memref_d.DeallocOp(buf)

                    func_d.ReturnOp([])

        (self.project_dir / "top.mlir").write_text(str(self.air_module), encoding="utf-8")
        (self.project_dir / "allo.mlir").write_text(str(self.allo_module), encoding="utf-8")

    def _lower_kernel_into_herd(
        self,
        kernel: allo_func_d.FuncOp,
        vmap: dict[Any, Any],
        ms2: IntegerAttr,
        keep_alive: set[Any],
    ):
        """Lower supported ops from an Allo kernel function into current insertion point.

        Parameters
        ----------
        keep_alive:
            Set of memrefs that must not be auto-deallocated here (e.g. local
            copies of kernel arguments). Any memref.alloc created in the kernel
            body without a matching dealloc will be deallocated at the end.
        """

        def map_val(v):
            return vmap.get(v, v)

        def map_memref_type(t: AlloMemRefType) -> MemRefType:
            elem_ty = _convert_allo_type_to_air(t.element_type)
            return MemRefType.get(tuple(t.shape), elem_ty, None, ms2)

        allocs_created: list[Any] = []
        allocs_deallocated: set[Any] = set()

        for blk in kernel.body.blocks:
            for op in blk.operations:
                opname = op.operation.name

                if opname == "func.return":
                    continue

                if isinstance(op, allo_arith_d.ConstantOp):
                    res_ty = _convert_allo_type_to_air(op.result.type)
                    if isinstance(op.value, AlloIntegerAttr):
                        v = int(op.value.value)
                    elif isinstance(op.value, AlloFloatAttr):
                        v = float(op.value.value)
                    else:
                        v = int(str(op.value))
                    vmap[op.result] = arith_d.ConstantOp(res_ty, v).result
                    continue

                if isinstance(op, allo_memref_d.AllocOp):
                    t = AlloMemRefType(op.result.type)
                    new_v = memref_d.AllocOp(map_memref_type(t), [], []).result
                    vmap[op.result] = new_v
                    allocs_created.append(new_v)
                    continue

                if isinstance(op, allo_memref_d.DeallocOp):
                    buf = map_val(op.operands[0])
                    memref_d.DeallocOp(buf)
                    allocs_deallocated.add(buf)
                    continue

                if isinstance(op, allo_memref_d.CopyOp):
                    memref_d.CopyOp(map_val(op.operands[0]), map_val(op.operands[1]))
                    continue

                if isinstance(op, allo_linalg_d.AddOp):
                    out = map_val(op.operands[-1])
                    elem_ty = MemRefType(out.type).element_type
                    new_op = linalg_d.AddOp([], [map_val(op.operands[0]), map_val(op.operands[1])], [out])
                    _build_linalg_add_region(new_op, elem_ty)
                    continue

                if isinstance(op, allo_linalg_d.MulOp):
                    out = map_val(op.operands[-1])
                    elem_ty = MemRefType(out.type).element_type
                    new_op = linalg_d.MulOp([], [map_val(op.operands[0]), map_val(op.operands[1])], [out])
                    _build_linalg_mul_region(new_op, elem_ty)
                    continue

                if isinstance(op, allo_linalg_d.FillOp):
                    out = map_val(op.operands[1])
                    elem_ty = MemRefType(out.type).element_type
                    new_op = linalg_d.FillOp([], [map_val(op.operands[0])], [out])
                    _build_linalg_fill_region(new_op, elem_ty)
                    continue

                if isinstance(op, allo_linalg_d.BroadcastOp):
                    out = map_val(op.operands[1])
                    elem_ty = MemRefType(out.type).element_type
                    dims: list[int] = []
                    if "dimensions" in op.attributes:
                        dims = _dense_i64_array_to_list(op.attributes["dimensions"])
                    new_op = linalg_d.BroadcastOp([], map_val(op.operands[0]), out, dims)
                    _build_linalg_broadcast_region(new_op, elem_ty)
                    continue

                raise NotImplementedError(
                    f"AIR backend: unsupported op in df.kernel lowering: {opname} ({type(op)})"
                )

        # Auto-deallocate kernel temporaries that the Allo kernel didn't dealloc.
        for buf in allocs_created:
            if buf in keep_alive:
                continue
            if buf in allocs_deallocated:
                continue
            memref_d.DeallocOp(buf)

    def __call__(self, *args):
        _call_prj(str(self.project_dir), self.output_idx, *args)


# -----------------------------------------------------------------------------
# Backend entry point
# -----------------------------------------------------------------------------


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
    """Entry point used by :meth:`allo.customize.Schedule.build(target='air')`."""

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
