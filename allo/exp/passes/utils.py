# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from ..utils import SymbolTable
from allo._mlir.ir import (
    Module,
    Location,
    InsertionPoint,
    BlockArgument,
    Operation,
    FlatSymbolRefAttr,
    StringAttr,
    AffineMap,
    AffineMapAttr,
    AffineConstantExpr,
)
from allo._mlir.dialects import affine as affine_d, allo as allo_d, func as func_d
from allo.memory import Layout, DTensor


def unroll_meta_for(module):
    def unroll(operations):
        for op in operations:
            for region in op.regions:
                for block in region.blocks:
                    unroll(block.operations)
            if isinstance(op, affine_d.AffineForOp):
                if (
                    "loop_type" in op.attributes
                    and op.attributes["loop_type"].value == "unroll"
                ):
                    allo_d.explicit_unroll(op)

    for func in module.body.operations:
        if isinstance(func, func_d.FuncOp):
            for block in func.body:
                unroll(block.operations)


def collect_symbol_refs_in_function(func_op: Operation):
    """
    Get symbol refs used in `func_op`. Symbols includes: global constant, streams, other 'function's
    """
    rs_symbols = defaultdict(list)
    kernel_symbols = defaultdict(list)

    def collect_recursive(operations):
        for op in operations:
            for attr in op.attributes.values():
                if isinstance(attr, FlatSymbolRefAttr):
                    if isinstance(op, func_d.CallOp):
                        kernel_symbols[attr.value].append(op)
                    else:
                        rs_symbols[attr.value].append(op)
            for region in op.regions:
                for block in region.blocks:
                    collect_recursive(block.operations)

    for block in func_op.body:
        collect_recursive(block.operations)
    return rs_symbols, kernel_symbols


@dataclass
class Unit:
    grid: dict
    dtensors: dict
    resources: dict = None
    works: dict = None
    top = None


def parse_namespace(unit_module):
    work_grids = {}
    dtensors = defaultdict(list)
    for func_block in unit_module.body:
        for op in func_block.operations:
            if isinstance(op, allo_d.GridMapOp):
                grid = list(op.grid)
                is_input = op.interfaces
                for block in op.body:
                    for sub_op in block.operations:
                        if isinstance(sub_op, func_d.CallOp):
                            work_grids[sub_op.callee.value] = grid
                            break
                for i, (buf, shard) in enumerate(zip(op.tensors, op.sharding)):
                    sharding = [
                        Layout.Shard(s.value) if s.value >= 0 else Layout.Replicate
                        for s in shard
                    ]
                    arg = BlockArgument(buf)
                    dtensor = DTensor(
                        mapping=grid,
                        shape=tuple(arg.type.shape),
                        dtype=arg.type.element_type,
                        spec_list=[Layout(sharding)],
                        tile_shape=tuple(sub_op.operands_[i].type.shape),
                        id_=arg.arg_number,
                    )
                    dtensors[sub_op.callee.value].append(dtensor)
                for i in range(len(op.tensors), len(sub_op.operands_)):
                    arg = BlockArgument(sub_op.operands_[i])
                    shape = tuple(arg.type.shape)
                    dtensor = DTensor(
                        mapping=grid,
                        shape=shape,
                        dtype=arg.type.element_type,
                        spec_list=[Layout([Layout.Replicate] * len(shape))],
                        tile_shape=shape,
                        id_=arg.arg_number,
                    )
                    dtensors[sub_op.callee.value].append(dtensor)
    return work_grids, dtensors


def parse_hierarchical_spmw_module(module):
    with module.context, Location.unknown():
        mod = Module.create()
    symbol_map = {}
    namespace = defaultdict(set[str])
    for op in module.body.operations:
        if "sym_name" in op.attributes:
            name = op.attributes["sym_name"].value
            symbol_map[name] = op
            namespace[SymbolTable.get_namespace(name)] = name
        if not isinstance(op, func_d.FuncOp):
            with InsertionPoint(mod.body), Location.unknown():
                op.clone()
    units = {}
    for name in namespace:
        work_grids, dtensors = parse_namespace(symbol_map[name])
        units[name] = Unit(work_grids, dtensors)

    return {
        "units": units,
        "symbols": symbol_map,
        "module": mod,
    }


def replace_stream_arrays(module):
    # Find all global stream arrays (len(shape) > 0)
    stream_arrays = {}
    for op in module.body.operations:
        if isinstance(op, allo_d.StreamGlobalOp) and len(op.shape) > 0:
            stream_arrays[op.sym_name.value] = op

    new_streams = {}

    def replace_recursive(operations, work_name):
        for op in operations:
            if isinstance(op, (allo_d.GlobalStreamGetOp, allo_d.GlobalStreamPutOp)):
                with op.context, Location.unknown():
                    stream_sym = op.attributes["global"].value
                    if stream_sym not in stream_arrays:
                        continue
                    allo_d.simplify_stream_affine_map(op)
                    aff_map = AffineMapAttr(op.map).value
                    assert aff_map.n_inputs == 0, "fail to resolve for now"
                    indices = [AffineConstantExpr(exp).value for exp in aff_map.results]
                    is_put = isinstance(op, allo_d.GlobalStreamPutOp)

                    new_name = f"{stream_sym}_{"_".join(map(str, indices))}"

                    # Create new stream Op if not exists
                    if new_name not in new_streams:
                        stream = stream_arrays[stream_sym]
                        new_op = allo_d.stream_global(
                            new_name, stream.element_type, [], ip=InsertionPoint(stream)
                        )
                        new_streams[new_name] = new_op

                    if is_put:
                        new_streams[new_name].attributes["source"] = StringAttr.get(
                            work_name
                        )
                    else:
                        new_streams[new_name].attributes["dest"] = StringAttr.get(
                            work_name
                        )

                    if is_put:
                        allo_d.put_stream_global(
                            FlatSymbolRefAttr.get(new_name),
                            [],
                            op.operands[-1],
                            AffineMapAttr.get(
                                AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
                            ),
                            ip=InsertionPoint(op),
                        )
                    else:
                        new_get = allo_d.GlobalStreamGetOp(
                            op.result.type,
                            FlatSymbolRefAttr.get(new_name),
                            [],
                            AffineMapAttr.get(
                                AffineMap.get(dim_count=0, symbol_count=0, exprs=[])
                            ),
                            ip=InsertionPoint(op),
                        )
                        op.result.replace_all_uses_with(new_get.result)

                    op.operation.erase()
            else:
                for region in op.regions:
                    for block in region.blocks:
                        replace_recursive(block.operations, work_name)

    for op in module.body.operations:
        if isinstance(op, func_d.FuncOp):
            name = op.sym_name.value
            for block in op.body:
                replace_recursive(block.operations, name)

    for op in stream_arrays.values():
        op.operation.erase()


def is_resource(op):
    if isinstance(op, func_d.FuncOp):
        return False
    return True
