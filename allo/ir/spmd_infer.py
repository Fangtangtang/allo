# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unused-argument, eval-used, redefined-variable-type, cell-var-from-loop

import ast
import copy
import sys
import os
import traceback
import inspect
import textwrap
import warnings
import sympy
import numpy as np

from .visitor import ASTVisitor, ASTContext, get_symbolic_expr
from .symbol_resolver import ASTResolver
from .types import (
    AlloType,
    Int,
    UInt,
    Fixed,
    UFixed,
    Index,
    uint1,
    int4,
    int8,
    int32,
    float16,
    float32,
    float64,
    Struct,
    Stream,
    stateful,
    ConstExpr,
)
from .typing_rule import get_typing_rule
from ..utils import (
    is_anywidth_int_type_and_not_np,
    get_bitwidth_from_type,
    handle_overflow,
    make_anywidth_numpy_array,
    np_supported_types,
    construct_kernel_name,
)
from ..memory import DTensor, Layout
from ..logging import print_error_message
from .utils import parse_ast, get_func_id_from_param_types, resolve_generic_types

# Using specific visit_stmt defined in this module
visit_stmt = None

class SPMDTypingInferer(ASTVisitor):
    def print_verbose(self, ctx: ASTContext, node: ast.AST):
        dtype = getattr(node, "dtype", "N/A")
        shape = getattr(node, "shape", "N/A")
        if isinstance(node, ast.Name):
            print("Name:", node.id, dtype, shape)
        else:
            print(node.__class__.__name__, dtype, shape)

    @staticmethod
    def visit_call_type(ctx: ASTContext, node: ast.expr):
        ty_cls = ASTResolver.resolve(node.func, ctx.global_vars)
        args = node.args

        if ty_cls is Fixed or ty_cls is UFixed:
            assert len(args) == 2
            assert isinstance(args[0], ast.Constant)
            assert isinstance(args[1], ast.Constant)
            dtype = ty_cls(
                ASTResolver.resolve_constant(args[0], ctx),
                ASTResolver.resolve_constant(args[1], ctx),
            )
        else:
            assert len(args) == 1
            dtype = ty_cls(ASTResolver.resolve_constant(args[0], ctx))
        return dtype

    @staticmethod
    def visit_type_hint(ctx: ASTContext, node: ast.AST):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Call):
                # e.g., a: UInt(16)[4]
                dtype = SPMDTypingInferer.visit_call_type(ctx, node.value)
            elif isinstance(node.value, ast.Subscript):
                # e.g., pipe: Stream[Ty, 4][4]
                base_type, base_shape, _ = SPMDTypingInferer.visit_type_hint(ctx, node.value)
                assert isinstance(base_type, Stream) and len(base_shape) == 0
                elts = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else [node.slice]
                )
                shape = tuple(ASTResolver.resolve(x, ctx.global_vars) for x in elts)
                assert all(
                    isinstance(x, (int)) for x in shape
                ), "stream array shape should be a compile time constant"
                return base_type, shape, None
            else:
                dtype = ASTResolver.resolve(node.value, ctx.global_vars)
            if dtype is Stream:
                # e.g., pipe: Stream[Ty, 4]
                assert (
                    isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2
                ), "Only support `ele_type` and `depth` for now"
                base_type, base_shape, _ = SPMDTypingInferer.visit_type_hint(
                    ctx, node.slice.elts[0]
                )
                depth = ASTResolver.resolve(node.slice.elts[1], ctx.global_vars)
                stream_dtype = Stream(dtype=base_type, shape=base_shape, depth=depth)
                shape = tuple()
                return stream_dtype, shape, None
            if dtype is ConstExpr:
                # e.g., a: ConstExpr[int32]
                base_type, base_shape, _ = SPMDTypingInferer.visit_type_hint(ctx, node.slice)
                assert len(base_shape) == 0, "ConstExpr only supports scalar types"
                const_dtype = copy.deepcopy(base_type)
                const_dtype.constexpr = True
                return const_dtype, tuple(), None
            assert dtype is not None, f"Unsupported type `{node.value.id}`"
            size = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            elts = size.elts if isinstance(size, ast.Tuple) else [size]
            shape = tuple(ASTResolver.resolve_constant(x, ctx) for x in elts)
            return (
                dtype,
                shape,
                Layout([Layout.Replicate] * len(shape)),
            )  # default layout
        if isinstance(node, ast.Name):
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            assert dtype is not None, f"Unsupported type `{node.id}`"
            return dtype, tuple(), None
        if isinstance(node, ast.Call):
            dtype = SPMDTypingInferer.visit_call_type(ctx, node)
            return dtype, tuple(), None
        if isinstance(node, ast.Constant):
            assert isinstance(node.value, str), "Only support string type annotation"
            tree = ast.parse(node.value)
            return SPMDTypingInferer.visit_type_hint(ctx, tree.body[0].value)
        if isinstance(node, ast.Attribute):
            # e.g., allo.ir.types.float32
            dtype = ASTResolver.resolve(node, ctx.global_vars)
            return dtype, tuple(), None
        if isinstance(node, ast.BinOp):
            # memory refinement
            # or, stateful variable
            # e.g., A: Ty[M] @ stateful
            dtype, shape, node_left_layout = SPMDTypingInferer.visit_type_hint(ctx, node.left)
            spec = ASTResolver.resolve(node.right, ctx.global_vars)
            if isinstance(spec, list):
                spec = Layout(spec)
            if spec is stateful:
                # Create a copy with stateful=True
                stateful_dtype = copy.deepcopy(dtype)
                stateful_dtype.stateful = True
                return stateful_dtype, shape, node_left_layout
            return dtype, shape, spec
        raise RuntimeError("Unsupported function argument type")

    @staticmethod
    def visit_broadcast(
        ctx: ASTContext,
        lhs_shape: list[int],
        rhs_shape: list[int],
        match_lhs: bool = False,
    ):
        if rhs_shape is None:
            return lhs_shape, [], []
        tmp_lhs_shape = list(lhs_shape)
        tmp_rhs_shape = list(rhs_shape)
        if match_lhs and len(tmp_lhs_shape) < len(tmp_rhs_shape):
            raise RuntimeError(f"Cannot broadcast {rhs_shape} to {lhs_shape}")
        # match larger shape
        lhs_dims, rhs_dims = set(), set()
        if len(tmp_lhs_shape) < len(tmp_rhs_shape):
            padded_dim = len(tmp_rhs_shape) - len(tmp_lhs_shape)
            tmp_lhs_shape = [1] * padded_dim + tmp_lhs_shape
            lhs_dims = set(range(padded_dim))
        elif len(tmp_lhs_shape) > len(tmp_rhs_shape):
            padded_dim = len(tmp_lhs_shape) - len(tmp_rhs_shape)
            tmp_rhs_shape = [1] * padded_dim + tmp_rhs_shape
            rhs_dims = set(range(padded_dim))
        # match shape
        # pylint: disable=consider-using-enumerate
        for i in range(len(tmp_lhs_shape)):
            if tmp_lhs_shape[i] == 1:
                tmp_lhs_shape[i] = tmp_rhs_shape[i]
                if tmp_rhs_shape[i] != 1:
                    if match_lhs:
                        raise RuntimeError(
                            f"Cannot broadcast {rhs_shape} to {lhs_shape}"
                        )
                    lhs_dims.add(i)
            elif tmp_rhs_shape[i] == 1:
                tmp_rhs_shape[i] = tmp_lhs_shape[i]
                if tmp_lhs_shape[i] != 1:
                    rhs_dims.add(i)
            else:
                assert (
                    tmp_lhs_shape[i] == tmp_rhs_shape[i]
                ), f"Shape mismatch, got {lhs_shape} and {rhs_shape}, and cannot be broadcasted"
        assert tmp_lhs_shape == tmp_rhs_shape
        return tuple(tmp_lhs_shape), list(lhs_dims), list(rhs_dims)

    @staticmethod
    def visit_general_binop(
        ctx: ASTContext, node: ast.AugAssign | ast.BinOp, lhs: ast.expr, rhs: ast.expr
    ):
        typing_rule = get_typing_rule(type(node.op), ctx.typing_rule_set)
        res_type = typing_rule(lhs.dtype, rhs.dtype)
        node.dtype = res_type
        final_shape, lhs_dims, rhs_dims = SPMDTypingInferer.visit_broadcast(
            ctx, lhs.shape, rhs_shape=None if rhs is None else rhs.shape
        )
        node.shape = final_shape
        node.dims = (lhs_dims, rhs_dims)
        if ctx.verbose:
            print(
                f"Broadcasted shape {lhs.shape} x {rhs.shape} -> {node.shape} for dims: {lhs_dims} & {rhs_dims}"
            )
        return node
    
    @staticmethod
    def visit_constant_tensor(
        ctx: ASTContext, node: ast.expr, np_values: np.array, dtype: AlloType
    ):
        dtype = str(dtype)
        if is_anywidth_int_type_and_not_np(dtype):
            bitwidth = get_bitwidth_from_type(dtype)
            if bitwidth <= 64:
                np_arr = handle_overflow(np_values, bitwidth, dtype)
                np_values = make_anywidth_numpy_array(np_arr, bitwidth)
        elif dtype in np_supported_types:
            target_np_type = np_supported_types[dtype]
            if np_values.dtype != target_np_type:
                # avoid changing the address of the original array
                np_values = np_values.astype(target_np_type)
        else:
            raise RuntimeError("Unsupported constant tensor element type")
        node.np_values = np_values
        node.shape = np_values.shape
        return node

    @staticmethod
    def visit_assignment_val(
        ctx: ASTContext,
        value: ast.expr,
        target_shape: list[int],
        target_dtype: AlloType,
    ):
        if isinstance(value, ast.List):
            assert target_shape is not None and target_dtype is not None
            values = compile(ast.Expression(value), "", "eval")
            # pylint: disable=eval-used
            values = np.array(eval(values, ctx.global_vars))
            assert (
                target_shape == values.shape
            ), f"Shape mismatch, got {target_shape} and {values.shape}"
            SPMDTypingInferer.visit_constant_tensor(ctx, value, values, dtype=target_dtype)
            value.dtype = target_dtype
        elif (
            isinstance(value, ast.Name)
            and value.id in ctx.global_vars
            and isinstance(ctx.global_vars[value.id], np.ndarray)
        ):
            assert target_shape is not None and target_dtype is not None
            assert (
                ctx.global_vars[value.id].shape == target_shape
            ), f"`{value.id}` shape mismatch, got {ctx.global_vars[value.id].shape} and {target_shape}"
            SPMDTypingInferer.visit_constant_tensor(
                ctx, value, ctx.global_vars[value.id], dtype=target_dtype
            )
            value.dtype = target_dtype
        elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            # Handle slicing of a constant numpy array, e.g., np_array[i]
            array_name = value.value.id
            if array_name in ctx.global_vars and isinstance(
                ctx.global_vars[array_name], np.ndarray
            ):
                assert target_shape is not None and target_dtype is not None
                np_array = ctx.global_vars[array_name]
                # Evaluate the slice at compile time
                slice_expr = compile(ast.Expression(value.slice), "", "eval")
                # pylint: disable=eval-used
                slice_val = eval(slice_expr, ctx.global_vars)
                # Extract the slice
                sliced_array = np_array[slice_val]
                # Ensure it's still a numpy array (scalar case)
                if not isinstance(sliced_array, np.ndarray):
                    sliced_array = np.array([sliced_array], dtype=np_array.dtype)
                assert (
                    sliced_array.shape == target_shape
                ), f"Slice shape mismatch, got {sliced_array.shape} and {target_shape}"
                SPMDTypingInferer.visit_constant_tensor(
                    ctx, value, sliced_array, dtype=target_dtype
                )
                value.dtype = target_dtype
            else:
                 visit_stmt(ctx, value)
        else:
             visit_stmt(ctx, value)
        return value

    @staticmethod
    def visit_FunctionDef(ctx: ASTContext, node: ast.FunctionDef):
        if ctx.top_func is not None:
             # Nested function def (kernels)
            old_ctx = ctx
            ctx = old_ctx.copy()
            ctx.scopes = old_ctx.scopes
            
            # Decorator handling (similar to TypeInferer but simplified)
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr == "kernel":
                            mapping, kernel_args = None, []
                            for kw in decorator.keywords:
                                if kw.arg == "mapping":
                                    mapping = eval(
                                        ast.unparse(kw.value),
                                        ctx.global_vars,
                                    )
                                elif kw.arg == "args":
                                    assert isinstance(kw.value, ast.List)
                                    kernel_args = kw.value.elts
                                    
                            assert mapping is not None, "Missing mapping in @df.kernel"
                            old_ctx.mapping = mapping
                            
                            # Handle args
                            for top_arg_name, arg in zip(kernel_args, node.args.args):
                                top_arg = ctx.get_symbol(name=top_arg_name.id)
                                dtype, shape, _ = SPMDTypingInferer.visit_type_hint(
                                    ctx, arg.annotation
                                )
                                assert top_arg.dtype == dtype and top_arg.shape == shape, \
                                    f"df.kernel argument {arg.arg} mismatch"
                                arg.top_arg = top_arg_name.id

                            # DIRECTLY visit the function definition without unrolling
                            # We need to setup a context where get_pid() might be called, 
                            SPMDTypingInferer._visit_FunctionDef_impl(ctx, node)
                            return node
        
        # If not a kernel or top-level, standard visitation
        SPMDTypingInferer._visit_FunctionDef_impl(ctx, node)
        return node

    @staticmethod
    def _visit_FunctionDef_impl(ctx: ASTContext, node: ast.FunctionDef):
        # Generic function handling 
        if hasattr(node, "type_params") and len(node.type_params) > 0:
             # ... (reuse logic if needed, or assume no generics for now? TypeInferer has it)
             pass

        with ctx.block_scope_guard():
            # Input types
            for arg in node.args.args:
                arg.dtype, arg.shape, arg.spec = SPMDTypingInferer.visit_type_hint(
                    ctx, arg.annotation
                )
                if hasattr(arg.dtype, "stateful") and arg.dtype.stateful:
                    raise RuntimeError(
                        f"Function parameter '{arg.arg}' cannot be Stateful."
                    )
                
                # DTensor creation (similar to TypeInferer)
                arg.dtensor = DTensor(
                    ctx.rank,
                    ctx.mapping,
                    arg.shape,
                    arg.dtype,
                    arg.spec,
                    name=arg.arg,
                    top_name=arg.arg if not hasattr(arg, "top_arg") else arg.top_arg,
                )
                
                # update shape
                arg.shape = arg.dtensor.get_local_shape()
                ctx.put_symbol(name=arg.arg, val=arg)

            func_name = node.name if ctx.func_id is None else f"{node.name}_{ctx.func_id}"
            
            # Return type
            if not ((isinstance(node.returns, ast.Constant) and node.returns.value is None) or node.returns is None):
                node.returns.dtype, node.returns.shape, node.returns.spec = (
                        SPMDTypingInferer.visit_type_hint(ctx, node.returns)
                    )
                ctx.put_symbol(name=func_name, val=node)

            ctx.top_func = node
            ctx.top_func_tree = node
            visit_stmts(ctx, node.body)
            
        ctx.global_vars[node.name] = node
        return node

    @staticmethod
    def visit_With(ctx: ASTContext, node: ast.With):
        assert len(node.items) == 1
        assert isinstance(node.items[0].context_expr, ast.Call)
        
        call_node = node.items[0].context_expr
        func_attr = call_node.func.attr if isinstance(call_node.func, ast.Attribute) else None
        
        if func_attr in {"meta_if", "meta_elif"}:
            # Check condition is constant
            cond_node = call_node.args[0]
            # Try to resolve constant
            cond_val = None
            try:
                cond_val = ASTResolver.resolve_constant(cond_node, ctx)
            except Exception:
                pass
            
            if cond_val is None:
                 # Try symbolic resolution
                 # Filter variables that are symbolic or constexpr
                 alive_var_names = ctx.get_alive_var_names()
                 filtered_var_names = set()
                 for name in alive_var_names:
                     if name in ctx.symbolic:
                         continue
                     sym = ctx.get_symbol(name)
                     def is_constexpr(dt):
                         if isinstance(dt, (list, tuple)):
                             return any(is_constexpr(t) for t in dt)
                         return getattr(dt, "constexpr", False)
                     if is_constexpr(sym.dtype):
                         continue
                     filtered_var_names.add(name)

                 try:
                     get_symbolic_expr(
                         copy.deepcopy(cond_node), 
                         ctx.symbolic, 
                         ctx.global_vars, 
                         filtered_var_names
                     )
                 except Exception as e:
                     raise RuntimeError(
                        f"Condition of {func_attr} must be a compile-time constant or valid symbolic expression."
                     ) from e
                 # It is symbolic, visit both branches if possible?
                 # Since we cannot determine control flow, we visit body.
                 # Validating body is type-safe is checking logic.
                 visit_stmts(ctx, node.body)
                 # Note: meta_else is separate node in AST? No, usually nested or separate items?
                 # Allo meta_if/else usage: `with meta_if: ... ; with meta_else: ...` 
                 # They are separate `With` nodes sequentially.
                 # So we just visit this one.
            else:
                 # Constant condition.
                 # We should technically only visit the active branch?
                 # But for type validity of the PROGRAM, arguably all code should be valid?
                 # Usually compiled out. If constant is False, maybe body is invalid type-wise but ignored?
                 # Let's visit anyway for safety checking? 
                 # Or follow infer.py logic: if unroll, only visit relevant. If not unroll, visit all.
                 visit_stmts(ctx, node.body)

        elif func_attr == "meta_else":
             # Just visit body. 
             # Logic is similar to meta_if symbolic case.
             visit_stmts(ctx, node.body)

        elif func_attr == "meta_for":
            # Check bounds are constant
            args = call_node.args
            for arg in args:
                try:
                    val = ASTResolver.resolve_constant(arg, ctx)
                    if val is None:
                        raise RuntimeError(
                             "Loop arguments of meta_for must be compile-time constant expressions."
                        )
                except Exception as e:
                    raise RuntimeError(
                         "Loop arguments of meta_for must be compile-time constant expressions."
                    ) from e
            
            # Only visit loop body once
            var_name = node.items[0].optional_vars.id
            
            # Create a loop variable as a Constant Index
            loop_var = ast.Name(id=var_name, ctx=ast.Load())
            loop_var.dtype = Index()
            loop_var.shape = ()
            # Mark as 'constexpr'
            loop_var.dtype.constexpr = True
            
            with ctx.block_scope_guard():
                ctx.put_symbol(var_name, loop_var)
                # Also support symbolic usage of loop var
                ctx.symbolic[var_name] = var_name 
                visit_stmts(ctx, node.body)
                
            return node
            
        else:
            raise RuntimeError(f"Unsupported context manager {func_attr}")

        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_Assign(ctx: ASTContext, node: ast.Assign):
        assert len(node.targets) == 1, "chained assignment not supported"
        target = node.targets[0]
        value = node.value
        
        # Handle get_pid symbolic registration
        if isinstance(value, ast.Call) and \
           isinstance(value.func, ast.Attribute) and \
           value.func.attr == "get_pid":
                # Register symbols for pid
                targets_list = []
                if isinstance(target, ast.Tuple):
                    targets_list = target.elts
                else:
                    targets_list = [target]
                
                # We assign symbolic names p0, p1, etc? 
                # Or just map the variable name to valid symbol?
                # get_symbolic_expr expects 'p0', 'p1' etc in mapping?
                # No, it expects mapping[name] = symbol.
                # allo uses p0, p1... for dims.
                # spmd_infer doesn't know dims if not unrolling? 
                # But we can just use "p{i}" as placeholder or just use variable name as symbol.
                for i, t in enumerate(targets_list):
                    if isinstance(t, ast.Name):
                         ctx.symbolic[t.id] = f"p{i}"

        # Evaluate RHS first
        rhs = visit_stmt(ctx, value)
        rhs_dtype = rhs.dtype
        rhs_shape = rhs.shape

        if isinstance(target, ast.Tuple):
             # Tuple assignment - simplified support for tuples in SPMD
             # Assuming elements are Names
             if isinstance(rhs, ast.Tuple) or isinstance(rhs, ast.Call):
                  # If RHS is a call that returns tuple (like min/max or user func with multiple returns)
                  pass 
             
             # For now, if we match logic from infer.py:
             # It iterates targets.
             pass
        
        if isinstance(target, ast.Name):
             target_ = ctx.get_symbol(target.id, allow_missing=True)
             if target_ is not None:
                  # Check for reassignment of constants
                  def is_constexpr(dt):
                      if isinstance(dt, (list, tuple)):
                          return any(is_constexpr(t) for t in dt)
                      return getattr(dt, "constexpr", False)

                  if is_constexpr(target_.dtype):
                       raise RuntimeError(f"Cannot reassign constant variable '{target.id}'")
                  # Standard reassignment
                  target.dtype, target.shape = target_.dtype, target_.shape
                  final_shape, lhs_dims, rhs_dims = SPMDTypingInferer.visit_broadcast(
                        ctx, target.shape, rhs_shape=rhs_shape, match_lhs=True
                  )
                  assert final_shape == target.shape, f"Shape mismatch: {final_shape} vs {target.shape}"
                  node.dims = target.dims = (lhs_dims, rhs_dims)
             else:
                  # New definition
                  ctx.put_symbol(name=target.id, val=target)
                  target.dtype = rhs_dtype
                  target.shape = rhs_shape
             
             node.dtype, node.shape = target.dtype, target.shape
        elif isinstance(target, ast.Subscript):
             # Assign to array element/slice
             lhs = visit_stmt(ctx, target)
             final_shape, lhs_dims, rhs_dims = SPMDTypingInferer.visit_broadcast(
                 ctx, lhs.shape, rhs_shape=rhs_shape, match_lhs=True
             )
             target.dtype, target.shape = lhs.dtype, lhs.shape
             node.dims = target.dims = (lhs_dims, rhs_dims)
        else:
             # Fallback to similar logic as infer.py if needed, or error
             pass
             
        return node

    @staticmethod
    def visit_AnnAssign(ctx: ASTContext, node: ast.AnnAssign):
        target_dtype, target_shape, spec = SPMDTypingInferer.visit_type_hint(
            ctx, node.annotation
        )
        assert isinstance(node.target, ast.Name), "target of AnnAssign must be Name"
        
        target_ = ctx.get_symbol(node.target.id, allow_missing=True)
        
        if target_ is not None:
             assert node.value is not None
             # Reassignment check
             if getattr(target_.dtype, "constexpr", False):
                  raise RuntimeError(f"Cannot reassign constant variable '{node.target.id}'")
             assert target_.dtype == target_dtype and target_.shape == target_shape
        
        # If ConstExpr
        if getattr(target_dtype, "constexpr", False):
             # Resolve value at compile time
             try:
                 val = ASTResolver.resolve(node.value, ctx.global_vars)
                 ctx.global_vars[node.target.id] = val
                 node.value.dtype = target_dtype
                 node.value.shape = target_shape
                 rhs = node.value
             except Exception as e:
                 raise RuntimeError(f"Failed to resolve ConstExpr '{node.target.id}'") from e
        else:
             rhs = SPMDTypingInferer.visit_assignment_val(
                 ctx, node.value, target_shape, target_dtype
             )

        if target_ is None: 
             ctx.put_symbol(name=node.target.id, val=node.target)
             
        node.target.dtype = node.dtype = target_dtype
        node.target.shape = node.shape = target_shape
        node.target.spec = node.spec = spec
        
        if rhs is not None:
            final_shape, lhs_dims, rhs_dims = SPMDTypingInferer.visit_broadcast(
                ctx, node.target.shape, rhs_shape=rhs.shape, match_lhs=True
            )
            assert final_shape == target_shape
            node.target.dims = node.dims = (lhs_dims, rhs_dims)
            
        return node
    
    @staticmethod
    def visit_AugAssign(ctx: ASTContext, node: ast.AugAssign):
        rhs = visit_stmt(ctx, node.value)
        if isinstance(node.target, ast.Subscript):
            lhs = visit_stmt(ctx, node.target)
        elif isinstance(node.target, ast.Name):
            lhs = ctx.get_symbol(node.target.id)
            assert lhs is not None
            node.target.dtype, node.target.shape = lhs.dtype, lhs.shape
        else:
            raise RuntimeError("Unsupported AugAssign")
        return SPMDTypingInferer.visit_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def visit_symbol(ctx: ASTContext, node: ast.expr):
        if isinstance(node, ast.Name):
            return sympy.symbols(node.id)
        if isinstance(node, ast.Constant):
            return sympy.Integer(node.value)
        if isinstance(node, ast.Attribute):
            assert isinstance(node.value, ast.Name)
            var = ctx.global_vars[node.value.id]
            if node.attr == "bits":
                return sympy.Integer(var.bits)
            if node.attr == "fracs":
                return sympy.Integer(var.fracs)
        if isinstance(node, ast.BinOp):
            lhs = SPMDTypingInferer.visit_symbol(ctx, node.left)
            rhs = SPMDTypingInferer.visit_symbol(ctx, node.right)
            op = {
                ast.Add: lambda l, r: l + r,
                ast.Sub: lambda l, r: l - r,
                ast.Mult: lambda l, r: l * r,
                ast.Div: lambda l, r: l / r,
                ast.FloorDiv: lambda l, r: l // r,
                ast.Mod: lambda l, r: l % r,
                ast.Pow: lambda l, r: l**r,
                ast.LShift: lambda l, r: l << r,
                ast.RShift: lambda l, r: l >> r,
                ast.BitOr: lambda l, r: l | r,
                ast.BitXor: lambda l, r: l ^ r,
                ast.BitAnd: lambda l, r: l & r,
            }.get(type(node.op))
            return op(lhs, rhs)
        raise TypeError(f"Unsupported symbol type {type(node)}")

    @staticmethod
    def visit_Subscript(ctx: ASTContext, node: ast.Subscript):
        value = visit_stmt(ctx, node.value)
        if len(value.shape) == 0 and isinstance(value.dtype, Struct):
            if not isinstance(node.slice, ast.Constant) or not isinstance(
                node.slice.value, str
            ):
                raise RuntimeError("Struct field access must use string literal")
            field = node.slice.value
            if field not in value.dtype.dtype_dict:
                raise RuntimeError(f"Field {field} not found in struct type")
            node.dtype = value.dtype.dtype_dict[field]
            node.shape = tuple()
            return node

        if len(value.shape) > 0:
            visit_stmt(ctx, node.slice)
            shape = []
            indices = ASTResolver.resolve_slice(node.slice, ctx)
            size = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
            elts = (
                size.elts
                if isinstance(size, ast.Tuple)
                else size.dims if isinstance(size, ast.ExtSlice) else [size]
            )
            access_dim = len(elts)
            total_dim = len(value.shape)
            if access_dim < total_dim:
                shape = value.shape[access_dim:]
            if isinstance(indices, tuple):
                indices = [indices]
            if isinstance(indices, list):
                for dim, index in enumerate(indices):
                    if isinstance(index, (list, tuple)):
                        lower = index[0] if index[0] is not None else 0
                        upper = (
                            index[1]
                            if index[1] is not None
                            else ctx.get_symbol(node.value.id).shape[dim]
                        )
                        step = (
                            index[2] if (len(index) > 2 and index[2] is not None) else 1
                        )
                        size = (upper - lower) // step
                        if size > 0:
                            shape.append(size)
            node.shape = tuple(shape)
            node.dtype = ctx.get_symbol(node.value.id).dtype
        elif len(value.shape) == 0 and isinstance(
            value.dtype, (Int, UInt)
        ):
            if isinstance(node.slice, (ast.Index, ast.Constant, ast.Name, ast.BinOp)):
                visit_stmt(ctx, node.slice)
                node.shape = tuple()
                node.dtype = uint1
            elif isinstance(node.slice, ast.Slice):
                lower_sym = SPMDTypingInferer.visit_symbol(ctx, node.slice.lower)
                upper_sym = SPMDTypingInferer.visit_symbol(ctx, node.slice.upper)
                if (
                    lower_sym is not None
                    and upper_sym is not None
                    and isinstance(upper_sym - lower_sym, sympy.core.numbers.Integer)
                ):
                    stride = int(upper_sym - lower_sym)
                    assert stride > 0, "upper bound must be greater than lower bound"
                    node.dtype = UInt(stride)
                else:
                    warnings.warn(
                        "Cannot infer the bitwidth of the slice, use UInt(32) as default"
                    )
                    node.dtype = UInt(32)
                lower = visit_stmt(ctx, node.slice.lower)
                upper = visit_stmt(ctx, node.slice.upper)
                node.shape = tuple()
            else:
                raise RuntimeError(f"Unsupported bit operation {node.slice}")
        else:
            raise RuntimeError("Can only access bit (slice) for integers")
        return node
    
    @staticmethod
    def visit_ExtSlice(ctx: ASTContext, node: ast.ExtSlice):
        stmts = visit_stmts(ctx, node.dims)
        node.shape = tuple()
        node.dtype = [stmt.dtype for stmt in stmts]
        return node

    @staticmethod
    def visit_Slice(ctx: ASTContext, node: ast.Slice):
        if node.lower is not None:
            visit_stmt(ctx, node.lower)
        if node.upper is not None:
            visit_stmt(ctx, node.upper)
        if node.step is not None:
            visit_stmt(ctx, node.step)
        node.shape = tuple()
        node.dtype = (Index(), Index(), Index())
        return node
        
    @staticmethod
    def visit_Index(ctx: ASTContext, node: ast.Index):
        value = visit_stmt(ctx, node.value)
        node.shape = value.shape
        node.dtype = value.dtype
        return node

    @staticmethod
    def visit_Tuple(ctx: ASTContext, node: ast.Tuple):
        visit_stmts(ctx, node.elts)
        node.shape = [elt.shape for elt in node.elts]
        node.dtype = [elt.dtype for elt in node.elts]
        return node

    @staticmethod
    def visit_Dict(ctx: ASTContext, node: ast.Dict):
        visit_stmts(ctx, node.keys)
        visit_stmts(ctx, node.values)
        node.dtype = Struct({k.value: v.dtype for k, v in zip(node.keys, node.values)})
        node.shape = ()
        return node

    @staticmethod
    def visit_Attribute(ctx: ASTContext, node: ast.Attribute):
        res = visit_stmt(ctx, node.value)
        if node.attr == "T":
            node.dtype = res.dtype
            node.shape = res.shape[::-1]
            return node
        if node.attr == "reverse":
            if not isinstance(res.dtype, (Int, UInt)):
                raise RuntimeError("Can only reverse integers")
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        if node.attr == "copy":
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        if node.attr in {"bits", "fracs"} and isinstance(res, ast.Name):
            node.dtype = res.dtype
            node.shape = res.shape
            return node
        raise RuntimeError(f"Unsupported attribute `{node.attr}`")

    @staticmethod
    def visit_Call(ctx: ASTContext, node: ast.Call):
        original_func_id = ctx.func_id
        if isinstance(node.func, ast.Name):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            obj = ASTResolver.resolve(node.func, ctx.global_vars)
            obj_name = node.func.attr
        elif isinstance(node.func, ast.Subscript):
            obj = ASTResolver.resolve(node.func.value, ctx.global_vars)
            assert obj is not None, "Unsupported function call"
            obj_name = obj.__name__
            ctx.global_vars[obj_name] = obj
            ctx.inst = ASTResolver.resolve_param_types(node.func.slice, ctx.global_vars)
            # func_id logic ...
            if ctx.func_id is None:
                func_id = get_func_id_from_param_types(ctx.inst)
                if func_id is None:
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    for key, value in func_dict.items():
                        if value == tuple(ctx.inst):
                            func_id = key
                            break
                    else:
                        func_id = len(func_dict) if len(func_dict) > 0 else None
                        func_dict[func_id] = tuple(ctx.inst)
                else:
                    ctx.inst.remove(func_id)
                    func_dict = ctx.func_name2id.setdefault(obj_name, {})
                    func_dict[func_id] = tuple(ctx.inst)
                ctx.func_id = func_id
        else:
            raise RuntimeError("Unsupported function call")

        if obj is None:
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "get_pid":
                     # CUSTOM: Handle get_pid returning constants
                     node.shape = (tuple(), tuple(), tuple())
                     idx_type = Index()
                     idx_type.constexpr = True
                     node.dtype = (idx_type, idx_type, idx_type)
                     return node
                
                if node.func.attr in {"T", "reverse"}:
                    assert len(node.args) == 0
                    attr = visit_stmt(ctx, node.func)
                    node.shape = attr.shape
                    node.dtype = attr.dtype
                elif node.func.attr == "put":
                    new_args = visit_stmts(ctx, node.args)
                    node.shape = tuple()
                    node.dtype = None
                    vid = (
                        node.func.value.id
                        if isinstance(node.func.value, ast.Name)
                        else node.func.value.value.id
                    )
                    # We skip symbolic loop unrolling hints logic from infer.py for now, or copy if needed.
                    val = ctx.get_symbol(vid)
                    node.func.value.shape = val.dtype.shape
                    node.func.value.dtype = val.dtype.dtype
                elif node.func.attr == "get":
                    vid = (
                        node.func.value.id
                        if isinstance(node.func.value, ast.Name)
                        else node.func.value.value.id
                    )
                    val = ctx.get_symbol(vid)
                    node.shape = val.dtype.shape
                    node.dtype = val.dtype.dtype
                    node.func.value.shape = tuple()
                    node.func.value.dtype = val.dtype
                elif node.func.attr == "bitcast":
                    visit_stmt(ctx, node.func.value)
                    node.shape = tuple()
                    if isinstance(node.func.value.dtype, (UInt, Int)):
                        if node.func.value.dtype.bits == 16:
                            node.dtype = float16
                        elif node.func.value.dtype.bits == 32:
                            node.dtype = float32
                        elif node.func.value.dtype.bits == 64:
                            node.dtype = float64
                        else:
                            raise RuntimeError(f"Unsupported bitwidth {node.func.value.dtype.bits}")
                    else:
                        node.dtype = UInt(node.func.value.dtype.bits)
                else:
                    raise RuntimeError(f"Unsupported function call or attribute method `.{node.func.attr}`")
            elif node.func.id in {"float", "int"}:
                assert len(node.args) == 1
                new_args = visit_stmts(ctx, node.args)
                node.shape = tuple()
                node.dtype = float32 if node.func.id == "float" else int32
            elif node.func.id in {"min", "max"}:
                assert len(node.args) == 2
                new_args = visit_stmts(ctx, node.args)
                typing_rule = get_typing_rule("minmax", ctx.typing_rule_set)
                res_type = typing_rule(new_args[0].dtype, new_args[1].dtype)
                node.dtype = res_type
                node.shape = new_args[0].shape
            else:
                 raise RuntimeError(f"Unsupported function call {node.func.id}")
            return node
        
        # ... Handle local imports and subfunctions ...
        # Simplified for brevity, assume library calls mostly
        
        if (
            obj.__module__.startswith("allo")
            and not obj.__module__.startswith("allo.library")
            and not obj.__module__.startswith("allo._mlir")
        ):
             # IP/ExternalModule handling
             # Allo library functions
             new_args = visit_stmts(ctx, node.args)
             if len(new_args) == 0:
                  if obj.__name__ == "get_pid":
                       node.shape = (tuple(), tuple(), tuple())
                       idx_type = Index()
                       idx_type.constexpr = True
                       node.dtype = (idx_type, idx_type, idx_type)
                       return node
                  node.shape = None; node.dtype = None
                  return node
             
             if all(len(arg.shape) == 0 for arg in new_args):
                 # element-wise
                 node.shape = tuple()
                 node.dtype = new_args[0].dtype
                 return node
             
             return SPMDTypingInferer.visit_library_op(ctx, node=node, op_name=obj.__name__, new_args=new_args)

        # User-defined subfunction
        if isinstance(obj, ast.FunctionDef):
             visit_stmts(ctx, node.args)
             pass
             
        visit_stmts(ctx, node.args)
        node.dtype = None
        node.shape = None
        ctx.func_id = original_func_id
        return node

    @staticmethod
    def visit_library_op(
        ctx: ASTContext, node: ast.Call, op_name: str, new_args: list[ast.AST]
    ):
        if op_name in {
            "exp",
            "softmax",
            "abs",
            "log",
            "add",
            "sub",
            "mul",
            "div",
            "relu",
            "copy",
        }:
            # Element-wise operation
            if op_name in {"add", "sub", "mul", "div"}:
                final_shape, lhs_dims, rhs_dims = SPMDTypingInferer.visit_broadcast(
                    ctx, new_args[0].shape, new_args[1].shape
                )
                node.dims = (lhs_dims, rhs_dims)
                node.shape = final_shape
            else:
                node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"matmul", "bmm", "linear", "conv2d", "sumpool", "maxpool"}:
            argAshape = new_args[0].shape
            argBshape = new_args[1].shape
            node.dtype = new_args[0].dtype
            if op_name == "conv2d":
                node.shape = (
                    argAshape[0],
                    argBshape[0],
                    argAshape[2] - argBshape[2] + 1,
                    argAshape[3] - argBshape[3] + 1,
                )
            elif op_name in {"maxpool", "sumpool"}:
                node.shape = (
                    argAshape[0],
                    argAshape[1],
                    argAshape[2] - argBshape[0] + 1,
                    argAshape[3] - argBshape[1] + 1,
                )
            elif op_name == "matmul":
                # FIXME (Shihan): for aie backend
                if not ctx.unroll and node.dtype == int4:
                    node.dtype = int8
                assert (
                    argAshape[-1] == argBshape[-2]
                ), f"The last dimension of the first input and the second last dimension of the second input must be the same, got {argAshape} and {argBshape}"
                node.shape = tuple(argAshape[:-1] + argBshape[-1:])
            elif op_name == "bmm":
                assert (
                    len(argAshape) == 3 and len(argBshape) == 3
                ), f"Only support batch matrix multiplication of two 3D inputs, got {len(argAshape)} and {len(argBshape)}"
                assert (
                    argAshape[2] == argBshape[1]
                ), f"The third dimension of the first input and the second dimension of the second input must be the same, got {argAshape} and {argBshape}"
                assert (
                    argAshape[0] == argBshape[0]
                ), f"The first dimension of the first input and the first dimension of the second input must be the same, got {argAshape} and {argBshape}"
                node.shape = (argAshape[0], argAshape[1], argBshape[2])
            elif op_name == "linear":
                # The weight parameter (i.e., `new_args[1]`) should be 2D, see:
                # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
                assert len(argBshape) == 2
                assert argAshape[-1] == argBshape[-1]
                # bias = True
                if len(new_args) == 3 and new_args[2] is not None:
                    assert argBshape[0] == new_args[2].shape[0]
                node.shape = argAshape[:-1] + argBshape[:-1]
            return node
        if op_name in {"transpose"}:
            assert (
                len(new_args) <= 2
            ), f"Only support zero/one extra argument for {op_name}"
            if len(new_args) == 1:
                node.shape = new_args[0].shape[::-1]
                node.dtype = new_args[0].dtype
            else:
                shape = new_args[0].shape
                axes = compile(ast.Expression(new_args[1]), "", "eval")
                # pylint: disable=eval-used
                axes = eval(axes)
                assert len(shape) == len(
                    axes
                ), f"Transpose shape mismatch, should provide the same number of dimensions as the input, got {len(shape)} and {axes}"
                new_shape = [shape[new_dim] for new_dim in axes]
                node.shape = tuple(new_shape)
                node.dtype = new_args[0].dtype
            return node
        if op_name in {"view"}:
            axes = compile(ast.Expression(new_args[1]), "", "eval")
            # pylint: disable=eval-used
            axes = eval(axes)
            node.shape = axes
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"layernorm", "gelu", "tril"}:
            node.shape = new_args[0].shape
            node.dtype = new_args[0].dtype
            return node
        if op_name in {"ones", "zeros"}:
            axes = compile(ast.Expression(new_args[0]), "", "eval")
            # pylint: disable=eval-used
            axes = eval(axes)
            node.shape = axes
            assert (
                node.keywords[0].arg == "dtype"
            ), f"Only support `dtype` keyword argument for {op_name}"
            dtype = node.keywords[0].value.id
            if dtype.startswith("int"):
                node.dtype = int32
            elif dtype.startswith("float"):
                node.dtype = float32
            else:
                raise RuntimeError(f"Unsupported dtype {dtype}")
            return node
        if op_name == "concat":
            axis = node.keywords[0].value.value
            if len(new_args[0].shape) != len(new_args[1].shape):
                raise RuntimeError(
                    f"Concatenation requires the same number of dimensions, got {len(new_args[0].shape)} and {len(new_args[1].shape)}"
                )

            for i, (shape1, shape2) in enumerate(
                zip(new_args[0].shape, new_args[1].shape)
            ):
                if i != axis and shape1 != shape2:
                    raise RuntimeError(
                        f"Concatenation requires the same shape except the concatenation axis {axis}, got {new_args[0].shape} and {new_args[1].shape}"
                    )
            shape = list(new_args[0].shape)
            shape[axis] += new_args[1].shape[axis]
            node.shape = tuple(shape)
            node.dtype = new_args[0].dtype
            return node
        raise RuntimeError(f"Unsupported linalg operation {op_name}")

    @staticmethod
    def visit_BinOp(ctx: ASTContext, node: ast.BinOp):
        lhs = visit_stmt(ctx, node.left)
        rhs = visit_stmt(ctx, node.right)
        return SPMDTypingInferer.visit_general_binop(ctx, node, lhs, rhs)

    @staticmethod
    def visit_UnaryOp(ctx: ASTContext, node: ast.UnaryOp):
        operand = visit_stmt(ctx, node.operand)
        node.shape = operand.shape
        if isinstance(operand.dtype, UInt):
            node.dtype = Int(operand.dtype.bits)
        else:
            node.dtype = operand.dtype
        return node
    
    @staticmethod
    def visit_Compare(ctx: ASTContext, node: ast.Compare):
        lhs = visit_stmt(ctx, node.left)
        assert len(node.comparators) == 1, "Only support one comparator for now"
        rhs = visit_stmt(ctx, node.comparators[0])
        typing_rule = get_typing_rule(type(node.ops[0]), ctx.typing_rule_set)
        operand_type = typing_rule(lhs.dtype, rhs.dtype)
        node.dtype = operand_type
        node.shape = tuple()
        return node

    @staticmethod
    def visit_BoolOp(ctx: ASTContext, node: ast.BoolOp):
        visit_stmts(ctx, node.values)
        node.dtype = uint1
        node.shape = tuple()
        return node

    @staticmethod
    def visit_IfExp(ctx: ASTContext, node: ast.IfExp):
        visit_stmt(ctx, node.test)
        visit_stmt(ctx, node.body)
        visit_stmt(ctx, node.orelse)
        typing_rule = get_typing_rule(ast.IfExp, ctx.typing_rule_set)
        res_type = typing_rule(node.body.dtype, node.orelse.dtype)
        node.dtype = res_type
        node.shape = node.body.shape
        return node

    @staticmethod
    def visit_If(ctx: ASTContext, node: ast.If):
        visit_stmt(ctx, node.test)
        with ctx.block_scope_guard():
            visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            with ctx.block_scope_guard():
                visit_stmts(ctx, node.orelse)
        node.dtype = None
        node.shape = None
        return node

    @staticmethod
    def visit_While(ctx: ASTContext, node: ast.While):
        visit_stmt(ctx, node.test)
        with ctx.block_scope_guard():
            visit_stmts(ctx, node.body)
        if len(node.orelse) > 0:
            raise RuntimeError(
                "'else' clause for 'while' not supported"
            )
        node.dtype = None
        node.shape = None
        return node
    
    @staticmethod
    def visit_Expr(ctx: ASTContext, node: ast.Expr):
        if isinstance(node.value, ast.Constant):
            node.dtype = None
            node.shape = None
            return node
        if isinstance(node.value, ast.Call):
            visit_stmt(ctx, node.value)
            node.dtype = None
            node.shape = None
            return node
        raise RuntimeError(f"Unsupported expression: {node.value}")

    @staticmethod
    def visit_Pass(ctx: ASTContext, node: ast.Pass):
        node.dtype = None
        node.shape = None
        return node
        
    @staticmethod
    def visit_Return(ctx: ASTContext, node: ast.Return):
        res = visit_stmt(ctx, node.value)
        node.dtype = res.dtype if res is not None else None
        node.shape = res.shape if res is not None else None
        return node

    @staticmethod
    def visit_Name(ctx: ASTContext, node: ast.Name):
        var = ctx.get_symbol(node.id, allow_missing=True)
        if var is not None:
            node.dtype = var.dtype
            node.shape = var.shape
            return node
        if node.id in ctx.global_vars:
            var = ctx.global_vars[node.id]
            if isinstance(var, int):
                node.dtype = int32
                node.shape = tuple()
            elif isinstance(var, float):
                node.dtype = float32
                node.shape = tuple()
            elif isinstance(var, AlloType):
                node.dtype = Index()
                node.shape = tuple()
            else:
                raise RuntimeError(f"Unsupported global variable `{node.id}`")
            return node
        raise RuntimeError(f"Unsupported Name `{node.id}`")

    @staticmethod
    def visit_Constant(ctx: ASTContext, node: ast.Constant):
        node.shape = tuple()
        if isinstance(node.value, int):
            node.dtype = int32
        elif isinstance(node.value, float):
            node.dtype = float32
        elif isinstance(node.value, str):
            node.dtype = str
        elif node.value is None:
            return ASTResolver.resolve_constant(node.value, ctx)
        else:
            raise RuntimeError("Unsupported constant type")
        return node
    
    @staticmethod
    def visit_all_for(ctx: ASTContext, node: ast.For):
        with ctx.block_scope_guard():
            # Set loop induction variables
            if isinstance(node.target, ast.Tuple):
                ivs = list(node.target.elts)
            else:
                ivs = [node.target]
            for iv in ivs:
                iv.shape = tuple()
                iv.dtype = Index()
                iv_ = ctx.get_symbol(iv.id, allow_missing=True)
                assert (
                    iv_ is None
                ), "Please choose a different name for the loop iterator."
                ctx.put_symbol(name=iv.id, val=iv)
            visit_stmts(ctx, node.iter.args)
            visit_stmts(ctx, node.body)
            node.shape = None
            node.dtype = None
        return node

    @staticmethod
    def visit_For(ctx: ASTContext, node: ast.For):
        if node.orelse:
            raise RuntimeError("'else' clause for 'for' not supported in Allo kernels")
        with ctx.loop_scope_guard():
            if isinstance(node.iter, ast.Call):
                obj = ASTResolver.resolve(node.iter.func, ctx.global_vars)
                if (
                    obj is None
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ) or (obj is not None and obj.__name__ in {"grid", "reduction"}):
                    return SPMDTypingInferer.visit_all_for(ctx, node)
            raise RuntimeError("Unsupported for loop")

    @staticmethod
    def visit_Module(ctx: ASTContext, node: ast.Module):
        for stmt in node.body:
             visit_stmt(ctx, stmt)
        node.dtype = None
        node.shape = None
        return node

visit_stmt = SPMDTypingInferer()

def visit_stmts(ctx: ASTContext, stmts: list[ast.expr]):
    results = []
    for stmt in stmts:
        try:
            results.append(visit_stmt(ctx, stmt))
        except Exception as e:
            print(f"{traceback.format_exc()}")
            print_error_message(str(e), stmt, ctx.top_func_tree)
            sys.exit(1)
    return results
