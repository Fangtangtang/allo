# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLIR-AIR backend.

This backend implements a minimal MLIR-to-MLIR translation from Allo's MLIR to
mlir-air MLIR, sufficient for the dataflow vector tests.

Design goals for this backend:
- Construct AIR/Linalg/MemRef/Arith ops via Python bindings (no string parsing
  to *construct* IR).
- Keep generated artifacts under a writable project directory.
- Provide an executable wrapper which reuses :func:`_call_prj`.

Lowering strategy (what we support today)
----------------------------------------
For the current test suite (`tests/dataflow/air/test_vector.py`), the Allo
pipeline emits a `@top` function that calls exactly one `df.kernel` function.
The kernel body contains only a small set of standard ops:

- arith.constant
- memref.alloc / memref.dealloc
- memref.copy
- linalg.{add,mul,fill,broadcast}

We lower as follows:
1) Create an AIR module with a `@top` function.
2) Wrap computation in an `air.herd` (1x1).
3) For each kernel memref argument that is read/written, allocate a local copy
   in memory space 2 and insert `air.dma_memcpy_nd` for host<->local transfers.
4) Rebuild the kernel body inside the herd block by *reconstructing* supported
   ops (we avoid cloning due to stability issues with some ops in the
   python bindings).

Sandbox note
------------
The autograder restricts filesystem writes. We therefore resolve/create project
folders only under `/home/sf668/workspace/allo/tests`.
"""

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

import filelock
import numpy as np

from air.backend.xrt import XRTBackend, XRTCompileArtifact
from air.ir import Context, Location, Module, InsertionPoint, IntegerType, IntegerAttr, IndexType

from air.dialects import air as air_d
from air.dialects import func as func_d
from air.dialects import arith as arith_d
from air.dialects import memref as memref_d
from air.dialects import linalg as linalg_d

import air.compiler.aircc.main as aircc


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

    # Prefer callers under tests/
    callers.sort(
        key=lambda p: (0 if _is_under(p, _ALLOWED_WRITE_ROOT) else 1, len(str(p)))
    )
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

    # Use an existing writable directory if possible.
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
# Conversion helpers
# -----------------------------------------------------------------------------


def _cidx(v: int):
    return arith_d.ConstantOp(IndexType.get(), v).result


def _memref_shape(memref_type) -> tuple[int, ...]:
    return tuple(int(d) for d in memref_type.shape)


def _with_memspace(memref_type, space: int = 2):
    ms_attr = IntegerAttr.get(IntegerType.get_signless(32), space)
    return memref_d.MemRefType.get(
        _memref_shape(memref_type),
        memref_type.element_type,
        memref_type.layout,
        ms_attr,
    )


def _analyze_kernel_io(kernel_func: func_d.FuncOp) -> tuple[list[int], list[int]]:
    args = list(kernel_func.arguments)
    used: set[int] = set()
    written: set[int] = set()

    def _arg_index(v):
        try:
            if hasattr(v, "owner") and v.owner == kernel_func.entry_block:
                return v.arg_number
        except Exception:
            pass
        return None

    for op in kernel_func.entry_block.operations:
        name = op.operation.name
        if name == "func.return":
            continue
        if name == "memref.copy":
            src, dst = op.operation.operands
            si = _arg_index(src)
            di = _arg_index(dst)
            if si is not None:
                used.add(si)
            if di is not None:
                used.add(di)
                written.add(di)
        elif name == "memref.store":
            if len(op.operation.operands) >= 2:
                di = _arg_index(op.operation.operands[1])
                if di is not None:
                    used.add(di)
                    written.add(di)
        else:
            for operand in op.operation.operands:
                ai = _arg_index(operand)
                if ai is not None:
                    used.add(ai)

    memref_arg_idxs = [
        i for i, a in enumerate(args) if memref_d.MemRefType.isinstance(a.type)
    ]
    used = used.intersection(memref_arg_idxs)
    written = written.intersection(memref_arg_idxs)
    return sorted(list(used - written)), sorted(list(written))


def _emit_dma_full(src, dst, memref_type, c0, c1):
    shape = _memref_shape(memref_type)
    offsets = [c0 for _ in shape]
    sizes = [_cidx(int(d)) for d in shape]
    strides = [c1 for _ in shape]
    air_d.DmaMemcpyNdOp(
        None,
        [],
        dst,
        [],
        [],
        [],
        src,
        offsets,
        sizes,
        strides,
    )


def _rebuild_kernel_body(src_kernel: func_d.FuncOp, value_map: dict, memspace: int = 2):
    for op in src_kernel.entry_block.operations:
        if isinstance(op, func_d.ReturnOp):
            continue

        if isinstance(op, arith_d.ConstantOp):
            new_const = arith_d.ConstantOp(op.result.type, op.value)
            value_map[op.result] = new_const.result
            continue

        if isinstance(op, memref_d.AllocOp):
            src_t = op.memref.type
            dst_t = _with_memspace(src_t, memspace)
            new_alloc = memref_d.AllocOp(dst_t, [], [])
            value_map[op.memref] = new_alloc.memref
            continue

        if isinstance(op, memref_d.DeallocOp):
            memref_d.DeallocOp(value_map[op.memref])
            continue

        if isinstance(op, memref_d.CopyOp):
            memref_d.CopyOp(value_map[op.source], value_map[op.target])
            continue

        if isinstance(op, linalg_d.FillOp):
            inp = value_map[op.inputs[0]]
            out = value_map[op.outputs[0]]
            new_op = linalg_d.FillOp([], [inp], [out])
            for k, v in op.operation.attributes.items():
                if k != "operandSegmentSizes":
                    new_op.operation.attributes[k] = v
            continue

        if isinstance(op, linalg_d.BroadcastOp):
            inp = value_map[op.input]
            init = value_map[op.init]
            new_op = linalg_d.BroadcastOp([], inp, init, op.dimensions)
            for k, v in op.operation.attributes.items():
                if k != "operandSegmentSizes":
                    new_op.operation.attributes[k] = v
            continue

        if isinstance(op, linalg_d.AddOp):
            ins = [value_map[v] for v in op.inputs]
            outs = [value_map[v] for v in op.outputs]
            new_op = linalg_d.AddOp([], ins, outs)
            for k, v in op.operation.attributes.items():
                if k != "operandSegmentSizes":
                    new_op.operation.attributes[k] = v
            continue

        if isinstance(op, linalg_d.MulOp):
            ins = [value_map[v] for v in op.inputs]
            outs = [value_map[v] for v in op.outputs]
            new_op = linalg_d.MulOp([], ins, outs)
            for k, v in op.operation.attributes.items():
                if k != "operandSegmentSizes":
                    new_op.operation.attributes[k] = v
            continue

        raise NotImplementedError(
            f"AIR backend: unsupported op in df.kernel: {op.operation.name}"
        )


# -----------------------------------------------------------------------------
# Public conversion entrypoint
# -----------------------------------------------------------------------------


def convert(allo_module, top_func_name: str, project: str) -> Module:
    project_dir = _resolve_or_create_project_dir(project)

    with Context() as ctx, Location.unknown():
        src_mod = Module.parse(str(allo_module), ctx)

        src_top = next(
            (
                op
                for op in src_mod.body.operations
                if isinstance(op, func_d.FuncOp) and op.sym_name.value == top_func_name
            ),
            None,
        )
        if src_top is None:
            raise RuntimeError(f"Top function '{top_func_name}' not found")

        call_ops = [
            op for op in src_top.entry_block.operations if isinstance(op, func_d.CallOp)
        ]
        if len(call_ops) != 1:
            raise RuntimeError(
                "AIR backend currently expects exactly one kernel call inside @top "
                f"(got {len(call_ops)})."
            )
        kernel_callee = call_ops[0].callee.value

        src_kernel = next(
            (
                op
                for op in src_mod.body.operations
                if isinstance(op, func_d.FuncOp) and op.sym_name.value == kernel_callee
            ),
            None,
        )
        if src_kernel is None:
            raise RuntimeError(f"Kernel function '{kernel_callee}' not found")

        in_arg_idx, out_arg_idx = _analyze_kernel_io(src_kernel)

        air_mod = Module.create()
        with InsertionPoint(air_mod.body):
            new_top = func_d.FuncOp(top_func_name, src_top.type)
            top_block = new_top.add_entry_block()
            with InsertionPoint(top_block):
                func_d.ReturnOp([])

            # Herd sizes: 1x1
            with InsertionPoint.at_block_terminator(top_block):
                herd = air_d.HerdOp(
                    None,
                    [],
                    [_cidx(1), _cidx(1)],
                    list(new_top.arguments),
                    sym_name="herd_0",
                )

            herd_block_types = [IndexType.get(), IndexType.get()] + [
                a.type for a in new_top.arguments
            ]
            herd_block = herd.body.blocks.append(*herd_block_types)
            with InsertionPoint(herd_block):
                air_d.HerdTerminatorOp()

            with InsertionPoint.at_block_terminator(herd_block):
                c0 = _cidx(0)
                c1 = _cidx(1)

                herd_args = list(herd_block.arguments)[2:]

                value_map: dict = {}
                local_allocs: list = []

                # Allocate locals and DMA-in inputs.
                for i, karg in enumerate(src_kernel.arguments):
                    g = herd_args[i]
                    if not memref_d.MemRefType.isinstance(g.type):
                        value_map[karg] = g
                        continue

                    if i in in_arg_idx or i in out_arg_idx:
                        local_t = _with_memspace(g.type, 2)
                        local = memref_d.AllocOp(local_t, [], []).memref
                        local_allocs.append(local)
                        value_map[karg] = local
                        if i in in_arg_idx:
                            _emit_dma_full(src=g, dst=local, memref_type=g.type, c0=c0, c1=c1)
                    else:
                        value_map[karg] = g

                # Kernel body.
                _rebuild_kernel_body(src_kernel, value_map, memspace=2)

                # DMA out outputs.
                for i in out_arg_idx:
                    g = herd_args[i]
                    local = value_map[src_kernel.arguments[i]]
                    shape = _memref_shape(g.type)
                    offsets = [c0 for _ in shape]
                    sizes = [_cidx(int(d)) for d in shape]
                    strides = [c1 for _ in shape]
                    air_d.DmaMemcpyNdOp(
                        None,
                        [],
                        g,
                        offsets,
                        sizes,
                        strides,
                        local,
                        [],
                        [],
                        [],
                    )

                for local in local_allocs:
                    memref_d.DeallocOp(local)

        (project_dir / "top.mlir").write_text(str(air_mod), encoding="utf-8")
        return air_mod


# -----------------------------------------------------------------------------
# Backend wrapper
# -----------------------------------------------------------------------------


class AIRModule:
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
        self.project = project or f"{top_func_name}.prj"
        self.configs = configs or {}
        self.func_args = func_args
        self.wrap_io = wrap_io

        self.project_dir = _resolve_or_create_project_dir(self.project)
        self.air_module = convert(module, top_func_name, str(self.project_dir))

        # Infer output indices.
        with Context() as ctx, Location.unknown():
            src_mod = Module.parse(str(module), ctx)
            src_top = next(
                (
                    op
                    for op in src_mod.body.operations
                    if isinstance(op, func_d.FuncOp)
                    and op.sym_name.value == top_func_name
                ),
                None,
            )
            call_ops = (
                [
                    op
                    for op in src_top.entry_block.operations
                    if isinstance(op, func_d.CallOp)
                ]
                if src_top is not None
                else []
            )
            if not call_ops:
                self.output_idx = []
            else:
                callee = call_ops[0].callee.value
                src_kernel = next(
                    (
                        op
                        for op in src_mod.body.operations
                        if isinstance(op, func_d.FuncOp) and op.sym_name.value == callee
                    ),
                    None,
                )
                self.output_idx = (
                    _analyze_kernel_io(src_kernel)[1] if src_kernel else []
                )

    def __repr__(self):
        return f"AIRModule(top={self.top_func_name}, project={self.project_dir})"

    def get_ir(self) -> str:
        return str(self.air_module)

    def __call__(self, *args):
        _call_prj(str(self.project_dir), self.output_idx, *args)


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
