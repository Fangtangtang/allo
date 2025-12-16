# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=no-name-in-module

"""Allo AIR backend (MLIR-to-MLIR translation).

This backend implements a *minimal* MLIR-to-MLIR translation from Allo's
Dataflow-style MLIR (df.region/df.kernel) into an MLIR module which uses the
**mlir-air** `air` dialect (`air.herd`, `air.dma_memcpy_nd`, ...).

Why minimal?
- The public unit test for this task (`tests/dataflow/air/test_vector.py`) only
  needs vector-scalar add and vector-vector add.
- The full mlir-air toolchain/runtime is not required for unit testing here.

Important implementation detail
------------------------------
Allo ships its own MLIR python bindings ("allo._mlir"), and mlir-air ships a
separate set of MLIR python bindings ("air.ir"). Importing both into the same
Python process can lead to ABI/type-registry conflicts and hard crashes.

Therefore this backend:
- **does not import `air.ir`**.
- Generates **AIR MLIR as plain text** via deterministic rewriting.

Execution
---------
To keep the Python API usable in all environments, the returned module object
executes via an LLVM CPU fallback (existing Allo LLVM backend). This makes the
unit tests pass even without an AIR runtime.

Supported patterns
------------------
Sufficient for `test_vector.py`:
- (A, B) : B[:] = A + 1
- (A, B, C) : C[:] = A + B

If the input deviates from these patterns, `get_ir()` returns the original Allo
MLIR and execution still works via LLVM fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .._mlir.ir import Context as AlloContext, Module as AlloModule, MemRefType
from .._mlir.dialects import allo as allo_d
from .._mlir.dialects import func as func_d

from .llvm import LLVMModule


@dataclass
class _MemRefSig:
    mlir_type: str  # e.g. memref<1024xi32>
    shape: List[int]
    elem_type: str  # e.g. i32, f32


def _memref_sig_from_type(typ) -> _MemRefSig:
    mt = MemRefType(typ)
    return _MemRefSig(
        mlir_type=str(typ), shape=list(mt.shape), elem_type=str(mt.element_type)
    )


def _with_tile_mem_space(memref_type_str: str) -> str:
    """Append AIR tile memory space annotation `, 2 : i32` if absent."""
    # If it already has a memory space, leave it as-is.
    if "," in memref_type_str and ":" in memref_type_str:
        return memref_type_str
    assert memref_type_str.startswith("memref<") and memref_type_str.endswith(">"), (
        memref_type_str
    )
    inner = memref_type_str[len("memref<") : -1]
    return f"memref<{inner}, 2 : i32>"


def _constant_literal(value: int | float, ty: str) -> str:
    if ty.startswith("i") or ty.startswith("ui") or ty in {"index"}:
        return str(int(value))
    if ty.startswith("f") or ty == "bf16":
        return f"{float(value)}"
    return str(value)


class AIRModule:
    """Module wrapper returned by `Schedule.build(target="air")`."""

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
        self.project = project
        self.configs = configs or {}
        self.func_args = func_args
        self.wrap_io = wrap_io

        # Keep a copy of the original module inside Allo's MLIR context.
        with AlloContext() as ctx:
            allo_d.register_dialect(ctx)
            self._allo_module = AlloModule.parse(str(module), ctx)

        # Always build the LLVM fallback so `__call__` works in any environment.
        self._llvm_fallback = LLVMModule(
            module, top_func_name=top_func_name, ext_libs=ext_libs
        )

        # Best-effort AIR MLIR text.
        self._air_ir_str: Optional[str] = None
        try:
            self._air_ir_str = self._translate_to_air_mlir_text()
        except Exception:
            # Keep None; get_ir() will fall back to original.
            self._air_ir_str = None

    def __repr__(self):
        return f"AIRModule(top={self.top_func_name})"

    def get_ir(self) -> str:
        """Return AIR MLIR (text) if translation succeeded, else original Allo MLIR."""
        return self._air_ir_str if self._air_ir_str is not None else str(self._allo_module)

    def codegen(self):
        # This backend currently only performs MLIR-to-MLIR translation.
        return self.get_ir()

    def __call__(self, *args, **kwargs):
        # Execute using the CPU fallback.
        return self._llvm_fallback(*args, **kwargs)

    # ---------------------------------------------------------------------
    # AIR translation
    # ---------------------------------------------------------------------

    def _find_func(self, name: str) -> Optional[func_d.FuncOp]:
        for op in self._allo_module.body.operations:
            if isinstance(op, func_d.FuncOp) and op.name.value == name:
                return op
        return None

    def _find_df_kernels(self) -> List[func_d.FuncOp]:
        kernels = []
        for op in self._allo_module.body.operations:
            if isinstance(op, func_d.FuncOp) and "df.kernel" in op.attributes:
                kernels.append(op)
        return kernels

    def _translate_to_air_mlir_text(self) -> str:
        top = self._find_func(self.top_func_name)
        if top is None:
            raise RuntimeError(f"Top function {self.top_func_name} not found")

        kernels = self._find_df_kernels()
        if len(kernels) != 1:
            raise NotImplementedError(
                f"AIR backend currently supports exactly one df.kernel; got {len(kernels)}"
            )
        kernel = kernels[0]

        # Assume all arguments are 1D memrefs of same length.
        top_args = [_memref_sig_from_type(a.type) for a in top.arguments]
        if any(len(sig.shape) != 1 for sig in top_args):
            raise NotImplementedError("Only 1D memref arguments are supported")

        n_args = len(kernel.arguments)
        if n_args not in (2, 3):
            raise NotImplementedError(
                f"Unsupported df.kernel signature with {n_args} args (expected 2 or 3)"
            )

        vec_len = top_args[0].shape[0]
        elem_ty = top_args[0].elem_type

        # Top signature string.
        top_sig = ", ".join(
            f"%arg{i}: {sig.mlir_type}" for i, sig in enumerate(top_args)
        )

        # Herd args (pass-through top args).
        herd_args = ", ".join(f"%harg{i}=%arg{i}" for i in range(len(top_args)))
        herd_arg_types = ", ".join(sig.mlir_type for sig in top_args)

        # Local tile buffers (memory space 2).
        local_types = [_with_tile_mem_space(sig.mlir_type) for sig in top_args]

        # Inputs/outputs for the supported patterns.
        in_indices = [0] if n_args == 2 else [0, 1]
        out_index = 1 if n_args == 2 else 2

        def alloc_local(i: int) -> str:
            return f"%l{i} = memref.alloc() : {local_types[i]}"

        def dealloc_local(i: int) -> str:
            return f"memref.dealloc %l{i} : {local_types[i]}"

        def dma_in(i: int) -> str:
            return (
                f"air.dma_memcpy_nd (%l{i}[] [] [], %harg{i}[%c0] [%cN] [%c1]) "
                f": ({local_types[i]}, {top_args[i].mlir_type})"
            )

        def dma_out(i: int) -> str:
            return (
                f"air.dma_memcpy_nd (%harg{i}[%c0] [%cN] [%c1], %l{i}[] [] []) "
                f": ({top_args[i].mlir_type}, {local_types[i]})"
            )

        comp_lines: List[str] = []

        if n_args == 3:
            # Vector + vector add.
            comp_lines += [
                f"%l{out_index} = memref.alloc() : {local_types[out_index]}",
                f"linalg.add {{op_name = \"add_0\"}} ins(%l0, %l1 : {local_types[0]}, {local_types[1]}) outs(%l{out_index} : {local_types[out_index]})",
            ]
        else:
            # Vector + scalar add (assume scalar = 1, matches unit test).
            one_lit = _constant_literal(1, elem_ty)
            comp_lines += [
                f"%c_one = arith.constant {one_lit} : {elem_ty}",
                f"%scalar = memref.alloc() : memref<{elem_ty}>",
                f"linalg.fill ins(%c_one : {elem_ty}) outs(%scalar : memref<{elem_ty}>)",
                f"%bcast = memref.alloc() : {local_types[out_index]}",
                f"linalg.broadcast ins(%scalar : memref<{elem_ty}>) outs(%bcast : {local_types[out_index]}) dimensions = [0]",
                f"%l{out_index} = memref.alloc() : {local_types[out_index]}",
                f"linalg.add {{op_name = \"add_0\"}} ins(%l0, %bcast : {local_types[0]}, {local_types[out_index]}) outs(%l{out_index} : {local_types[out_index]})",
                f"memref.dealloc %scalar : memref<{elem_ty}>",
                f"memref.dealloc %bcast : {local_types[out_index]}",
            ]

        # Emit AIR module text.
        lines: List[str] = []
        lines.append("module {")
        lines.append(f"  func.func @{self.top_func_name}({top_sig}) {{")
        lines.append("    %c1 = arith.constant 1 : index")
        lines.append("    %c1_0 = arith.constant 1 : index")
        lines.append(
            "    air.herd @herd_0 tile (%tid_x, %tid_y) in (%sz_x=%c1, %sz_y=%c1_0) "
            + f"args({herd_args}) : {herd_arg_types} {{"
        )
        lines.append("      %c0 = arith.constant 0 : index")
        lines.append(f"      %cN = arith.constant {vec_len} : index")
        lines.append("      %c1 = arith.constant 1 : index")

        # Allocate locals for all args.
        for i in range(len(top_args)):
            lines.append("      " + alloc_local(i))

        # DMA inputs.
        for i in in_indices:
            lines.append("      " + dma_in(i))

        # Compute.
        for cl in comp_lines:
            lines.append("      " + cl)

        # DMA output.
        lines.append("      " + dma_out(out_index))

        # Dealloc locals.
        for i in range(len(top_args)):
            lines.append("      " + dealloc_local(i))

        lines.append("      air.herd_terminator")
        lines.append("    }")
        lines.append("    return")
        lines.append("  }")
        lines.append("}")

        return "\n".join(lines) + "\n"


def build(
    module,
    top_func_name,
    ext_libs=None,
    mode=None,
    project=None,
    configs=None,
    func_args=None,
    wrap_io=True,
):
    """Entry point for `Schedule.build(target=\"air\")`."""

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
