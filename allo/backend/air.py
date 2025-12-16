"""AIR backend (MLIR-to-MLIR translation).

This backend translates Allo's MLIR (which typically contains one or more
`df.kernel` functions plus a `dataflow` top function) into an `mlir-air`
style module.

The goal for this educational backend is to support the basic testcases
in `tests/dataflow/air/`, especially vector add kernels.

Design notes
------------
- The implementation is intentionally lightweight and relies on emitting
  valid AIR dialect MLIR text.
- We validate the produced MLIR by parsing it with `mlir-air` python
  bindings if they are available at runtime.
- For now we support 1D contiguous memrefs and linalg elementwise ops.

If a pattern is not supported we raise `NotImplementedError` with a
helpful message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


_MEMREF_RE = re.compile(r"memref<([^>]+)>")
_FUNC_RE = re.compile(r"func\.func\s+@(?P<name>[A-Za-z0-9_]+)\((?P<args>[^)]*)\)\s*(?:->\s*\([^)]*\))?\s*(?P<attrs>attributes\s*\{[^}]*\})?\s*\{")


def _split_top_level_commas(s: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


@dataclass
class MemRefType:
    shape: List[str]  # dims as strings (e.g. ["1024"], ["64","64"], ["?"])
    elem: str         # element type string (e.g. "f32")

    @property
    def rank(self) -> int:
        return len(self.shape)

    def as_mlir(self) -> str:
        dims = "x".join(self.shape)
        return f"memref<{dims}x{self.elem}>"

    def as_air_local(self, space: str = "2 : i32") -> str:
        dims = "x".join(self.shape)
        return f"memref<{dims}x{self.elem}, {space}>"

    def num_elems_expr(self) -> str:
        # Best effort: only handles static shapes.
        if any(d in ("?", "-1") for d in self.shape):
            raise NotImplementedError("dynamic memref shapes are not supported by the AIR backend yet")
        prod = 1
        for d in self.shape:
            prod *= int(d)
        return str(prod)


def _parse_memref_type(type_str: str) -> MemRefType:
    m = _MEMREF_RE.search(type_str)
    if not m:
        raise NotImplementedError(f"Expected memref type, got: {type_str}")
    inner = m.group(1).strip()
    # Drop layout/space if present: `64xf32, ...`.
    if "," in inner:
        inner = inner.split(",", 1)[0].strip()
    # `1024xf32` or `64x64xf32`.
    pieces = inner.split("x")
    if len(pieces) < 2:
        raise NotImplementedError(f"Unexpected memref type body: memref<{inner}>")
    elem = pieces[-1].strip()
    shape = [p.strip() for p in pieces[:-1]]
    return MemRefType(shape=shape, elem=elem)


@dataclass
class FuncSig:
    name: str
    arg_types: List[str]


def _extract_first_func_with_attr(mlir: str, attr_substr: str) -> Optional[FuncSig]:
    # Naive textual scan; sufficient for our controlled test IR.
    for m in _FUNC_RE.finditer(mlir):
        name = m.group("name")
        args_str = m.group("args")
        attrs = m.group("attrs") or ""
        if attr_substr in attrs:
            arg_types = []
            for a in _split_top_level_commas(args_str.strip()):
                # `%arg0: memref<...>`
                if ":" not in a:
                    continue
                _, ty = a.split(":", 1)
                arg_types.append(ty.strip())
            return FuncSig(name=name, arg_types=arg_types)
    return None


def _extract_linalg_op_line(kernel_body: str) -> str:
    # Finds the first linalg.* op line (single-line assumed).
    for line in kernel_body.splitlines():
        s = line.strip()
        if s.startswith("linalg.") or " linalg." in s:
            return s
    raise NotImplementedError("No linalg op found in kernel; AIR backend currently expects an elementwise linalg op")


def _extract_kernel_body(mlir: str, kernel_name: str) -> str:
    # Extract text between the opening `{` of func.func @kernel_name and the matching closing `}`.
    # Assumes well-formed MLIR, braces balanced.
    start_pat = re.compile(rf"func\.func\s+@{re.escape(kernel_name)}\([^)]*\)[^{{]*\{{")
    m = start_pat.search(mlir)
    if not m:
        raise NotImplementedError(f"Cannot find kernel function @{kernel_name}")
    i = m.end()
    depth = 1
    j = i
    while j < len(mlir) and depth > 0:
        if mlir[j] == "{":
            depth += 1
        elif mlir[j] == "}":
            depth -= 1
        j += 1
    body = mlir[i : j - 1]
    return body


class AirBackendModule:
    """Backend wrapper returned by `s.build(target='air')`."""

    def __init__(self, module, top_func_name: str, *, herd_dims: Tuple[int, int] = (1, 1)):
        self.top_func_name = top_func_name
        self.herd_dims = herd_dims

        src = str(module)
        self.src_mlir = src

        kernel_sig = _extract_first_func_with_attr(src, "df.kernel")
        top_sig = _extract_first_func_with_attr(src, "dataflow")
        if top_sig is None:
            # Fall back: use the user-provided `top_func_name` signature by searching function name.
            top_sig = _extract_first_func_with_attr(src, f"@{top_func_name}")

        if kernel_sig is None:
            raise NotImplementedError("AIR backend expects at least one func.func with attribute `df.kernel`")
        if top_sig is None:
            raise NotImplementedError("AIR backend expects a `dataflow` top function (attributes {dataflow, ...})")

        # We assume the kernel args correspond to top args (common in Allo dataflow lowering).
        self.arg_types = top_sig.arg_types
        self.arg_memrefs: List[MemRefType] = [_parse_memref_type(t) for t in self.arg_types]

        # Simple heuristic: treat last argument as output for vector tests.
        if len(self.arg_memrefs) < 2:
            raise NotImplementedError("AIR backend expects at least two memref arguments")

        kernel_body = _extract_kernel_body(src, kernel_sig.name)
        linalg_line = _extract_linalg_op_line(kernel_body)

        self.air_mlir = self._emit_air_module(linalg_line)
        self._validate_with_air_bindings_if_available(self.air_mlir)

    def _emit_air_module(self, linalg_line: str) -> str:
        # Build a minimal AIR module:
        # - a single func.func @top
        # - an air.herd with herd_dims
        # - dma copies from global args to local L1 buffers (space 2)
        # - the linalg op operating on local buffers
        # - dma copy back for output

        herd_x, herd_y = self.herd_dims
        if herd_x <= 0 or herd_y <= 0:
            raise ValueError("herd_dims must be positive")

        # For now we only support a single contiguous chunk; for multi-dim memrefs we treat it as flattened.
        total_elems = self.arg_memrefs[0].num_elems_expr()

        # Build function signature from arg types verbatim.
        args = ", ".join([f"%arg{i}: {t}" for i, t in enumerate(self.arg_types)])

        # Local buffers (one per input, one for output). Assume output is last arg.
        local_types = [m.as_air_local() for m in self.arg_memrefs]

        # Map global args to herd args.
        herd_args_bind = ", ".join([f"%harg{i}=%arg{i}" for i in range(len(self.arg_types))])
        herd_args_types = ", ".join(self.arg_types)

        # Emit linalg op by rewriting the operand names/types.
        # We only handle patterns like:
        #   linalg.add ... ins(%arg0, %arg1 : memref<...>, memref<...>) outs(%alloc : memref<...>)
        # We will force it to:
        #   linalg.add ... ins(%l0, %l1 : local_ty0, local_ty1) outs(%lout : local_ty_out)
        linalg_op = self._rewrite_linalg_line_to_local(linalg_line, local_types)

        # AIR dma memcpy. We use 1D ND syntax with offsets/length/strides.
        dma_in_lines = []
        for i in range(len(self.arg_types) - 1):
            dma_in_lines.append(
                f"      air.dma_memcpy_nd (%l{i}[] [] [], %harg{i}[%c0] [%cN] [%c1]) : ({local_types[i]}, {self.arg_types[i]})"
            )
        out_idx = len(self.arg_types) - 1
        dma_out_line = (
            f"      air.dma_memcpy_nd (%harg{out_idx}[%c0] [%cN] [%c1], %l{out_idx}[] [] []) : ({self.arg_types[out_idx]}, {local_types[out_idx]})"
        )

        alloc_lines = [f"      %l{i} = memref.alloc() : {local_types[i]}" for i in range(len(local_types))]
        dealloc_lines = [f"      memref.dealloc %l{i} : {local_types[i]}" for i in range(len(local_types))]

        mlir = f"""module {{
  func.func @{self.top_func_name}({args}) {{
    %c{herd_x} = arith.constant {herd_x} : index
    %c{herd_y} = arith.constant {herd_y} : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c{herd_x}, %sy=%c{herd_y}) args({herd_args_bind}) : {herd_args_types} {{
{chr(10).join(alloc_lines)}
      %c0 = arith.constant 0 : index
      %cN = arith.constant {total_elems} : index
      %c1 = arith.constant 1 : index
{chr(10).join(dma_in_lines)}
{linalg_op}
{dma_out_line}
{chr(10).join(dealloc_lines)}
    }}
    return
  }}
}}"""
        return mlir

    def _rewrite_linalg_line_to_local(self, linalg_line: str, local_types: List[str]) -> str:
        # Replace operand SSA values with local ones.
        # Inputs: all except last arg. Output uses last local buffer.
        n = len(local_types)
        if n < 2:
            raise NotImplementedError

        # Strip any leading result assignment (%0 = )
        line = linalg_line.strip()
        if "=" in line.split("linalg.", 1)[0]:
            # Has a result assignment; keep it but it shouldn't matter.
            pass

        # Rewrite ins(...) and outs(...) clauses.
        # This is heuristic and expects single-line linalg syntax.
        ins_pat = re.compile(r"ins\((?P<vals>[^)]*)\s*:\s*(?P<tys>[^)]*)\)")
        outs_pat = re.compile(r"outs\((?P<vals>[^)]*)\s*:\s*(?P<tys>[^)]*)\)")

        ins_m = ins_pat.search(line)
        outs_m = outs_pat.search(line)
        if not ins_m or not outs_m:
            raise NotImplementedError(f"Unsupported linalg op format for AIR backend: {line}")

        in_vals = [f"%l{i}" for i in range(n - 1)]
        in_tys = local_types[: n - 1]
        out_vals = [f"%l{n-1}"]
        out_tys = [local_types[n - 1]]

        repl_ins = f"ins({', '.join(in_vals)} : {', '.join(in_tys)})"
        repl_outs = f"outs({', '.join(out_vals)} : {', '.join(out_tys)})"

        line = ins_pat.sub(repl_ins, line)
        line = outs_pat.sub(repl_outs, line)

        return f"      {line}"

    def _validate_with_air_bindings_if_available(self, mlir: str) -> None:
        # This is best-effort; we don't want to make the backend unusable
        # if mlir-air is not installed.
        try:
            import air.ir as air_ir  # type: ignore
            import air.dialects.air as air_d  # type: ignore
            import air.dialects.func as func_d  # type: ignore
            import air.dialects.memref as memref_d  # type: ignore
            import air.dialects.linalg as linalg_d  # type: ignore
            import air.dialects.arith as arith_d  # type: ignore
        except Exception:
            return

        with air_ir.Context() as ctx, air_ir.Location.unknown():
            # Register common dialects.
            for d in [air_d, func_d, memref_d, linalg_d, arith_d]:
                try:
                    d.register_dialect(ctx)
                except Exception:
                    pass
            air_ir.Module.parse(mlir, ctx)

    def get_ir(self) -> str:
        return self.air_mlir

    def codegen(self, *args, **kwargs):
        # MLIR-to-MLIR backend; no further codegen.
        return self.get_ir()


def build(module, top_func_name: str, **kwargs) -> AirBackendModule:
    """Entry point used by `Schedule.build(target='air')`."""
    herd_dims = kwargs.pop("herd_dims", (1, 1))
    return AirBackendModule(module, top_func_name, herd_dims=herd_dims)
