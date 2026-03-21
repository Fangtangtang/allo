# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from .utils import (
    unroll_meta_for,
    parse_hierarchical_spmw_module,
    is_resource,
    collect_symbol_refs_in_function,
    replace_stream_arrays,
    Unit,
)
import allo._mlir.extras.dialects.func as func
import allo._mlir.extras.attr as attr
from allo._mlir.passmanager import PassManager as mlir_pass_manager
from allo.utils import construct_kernel_name as construct_name
import allo._mlir.extras.types as mlir_types
from allo._mlir.ir import (
    Module,
    Location,
    InsertionPoint,
    StringAttr,
    FlatSymbolRefAttr,
    UnitAttr,
)
from allo._mlir.dialects import (
    allo as allo_d,
    arith as arith_d,
    func as func_d,
)


def instantiate_for_hls(module, top_name):
    parsed = parse_hierarchical_spmw_module(module)

    mod = parsed["module"]
    symbol_map = parsed["symbols"]
    units: dict[str, Unit] = parsed["units"]
    instance_copies = {}

    with module.context, Location.unknown():
        copies = Module.create()
        with InsertionPoint(copies.body):
            for name in units.keys():  # units can be called in work
                f = func.signature(symbol_map[name])
            for op in mod.body.operations:
                if is_resource(op):
                    op.clone()
            for unit in units.values():
                for orig_func_name, grid_shape in unit.grid.items():
                    orig_func = symbol_map[orig_func_name]
                    for dim in np.ndindex(*grid_shape):
                        func_ = orig_func.clone()
                        function_name = construct_name(orig_func_name, dim)
                        func_.sym_name = StringAttr.get(function_name)
                        instance_copies[function_name] = func_
                        # meta data
                        for func_block in func_.body:
                            for op in func_block.operations:
                                assert isinstance(op, func_d.CallOp)
                                assert op.callee.value.endswith(
                                    "get_wid"
                                )  # FIXME: parsing
                                for res, num in zip(op.results, dim):
                                    const_op = arith_d.ConstantOp(
                                        mlir_types.index(), num, ip=InsertionPoint(op)
                                    )
                                    res.replace_all_uses_with(const_op.result)
                                op.erase()
                                break
                            break

        unroll_meta_for(copies)
        pipeline = "builtin.module(lower-memcopy-ops, canonicalize)"
        mlir_pass_manager.parse(pipeline).run(copies.operation)
        replace_stream_arrays(copies)

    counter = {k: 0 for k in units.keys()}
    # construct units
    for name, unit in units.items():
        top_module = symbol_map[name]
        resources = {}
        for k, v in symbol_map.items():
            if k.startswith(name) and is_resource(v):
                resources[k] = v

        unit.resources = {}
        unit.works = {}
        with InsertionPoint(mod.body), Location.unknown():
            top_func = func_d.FuncOp(name, top_module.type)
            unit.top = top_func
            entry_block = top_func.add_entry_block()
            attr.copy_attr(
                top_module, top_func, {"itypes", "otypes"}, allow_missing=False
            )
            entry_ip = InsertionPoint.at_block_begin(entry_block)
            if top_name == name:
                top_func.attributes["dataflow"] = UnitAttr.get()

            # move resource to local
            with entry_ip:
                for name, op in resources.items():
                    if isinstance(op, allo_d.StreamGlobalOp):
                        shape = list(op.shape)
                        if len(shape) == 0:
                            names = [name]
                        else:
                            names = [
                                construct_name(name, dim) for dim in np.ndindex(*shape)
                            ]
                        for n in names:
                            stream = allo_d.StreamConstructOp(op.element_type.value)
                            unit.resources[n] = stream.result
                    else:
                        raise NotImplementedError

        with InsertionPoint.at_block_begin(mod.body), Location.unknown():
            # call ops
            dtensors = unit.dtensors
            for orig_func_name, grid_shape in unit.grid.items():
                tensors = dtensors[orig_func_name]  # argument passing
                for dim in np.ndindex(*grid_shape):
                    function_name = construct_name(orig_func_name, dim)
                    instance = instance_copies[function_name]
                    # [NOTE]: backend do not support `.` in function names, backend use `startswith` to identify top module
                    hls_function_name = f"_{function_name.replace(".", "_")}"
                    symbols, kernels = collect_symbol_refs_in_function(instance)
                    for k in kernels:
                        if k in counter:
                            counter[k] += len(kernels[k])
                    symbol_names = sorted(symbols.keys())
                    itypes = instance.type.inputs
                    base_inputs = len(itypes)
                    assert "itypes" in instance.attributes
                    itype_hints = instance.attributes["itypes"].value
                    args = []
                    for t in tensors:
                        assert t.tile_shape == t.shape  # hls do not support sharding
                        args.append(top_func.arguments[t.global_id])
                    for symbol in symbol_names:
                        resource_op = unit.resources[symbol]
                        args.append(resource_op)
                        itypes.append(resource_op.type)
                        itype_hints += "_"  # FIXME
                    new_func = func.function(
                        hls_function_name,
                        itypes,
                        [],
                        itype_hints=itype_hints,
                        otype_hints="",
                    )
                    final_op = func_d.ReturnOp(
                        [], ip=InsertionPoint(new_func.entry_block)
                    )
                    # update arguments
                    for i, arg in enumerate(instance.arguments):
                        arg.replace_all_uses_with(new_func.arguments[i])
                    for op in instance.entry_block.operations:
                        op.operation.move_before(final_op)
                    final_op.erase()
                    # update resource usage
                    for i, symbol in enumerate(symbol_names):
                        ops = symbols[symbol]
                        for op in ops:
                            if isinstance(op, allo_d.GlobalStreamGetOp):
                                new_op = allo_d.StreamGetOp(
                                    op.result.type,
                                    new_func.arguments[i + base_inputs],
                                    [],
                                    ip=InsertionPoint(op),
                                )
                                op.result.replace_all_uses_with(new_op.result)
                                op.erase()
                            elif isinstance(op, allo_d.GlobalStreamPutOp):
                                allo_d.StreamPutOp(
                                    new_func.arguments[i + base_inputs],
                                    [],
                                    op.data,
                                    ip=InsertionPoint(op),
                                )
                                op.erase()
                            else:
                                print(op)
                                raise NotImplementedError
                    # call the function in top module
                    unit.works[hls_function_name] = new_func
                    func_d.CallOp(
                        [],
                        FlatSymbolRefAttr.get(hls_function_name),
                        args,
                        ip=entry_ip,
                    )
            func_d.ReturnOp([], ip=entry_ip)

    for op in mod.body.operations:
        if isinstance(op, allo_d.StreamGlobalOp):
            op.erase()

    if all(v <= 1 for v in counter.values()):
        return mod

    return instantiate_callee(mod, units, top_name)


def instantiate_callee(orig_module, units, top_name):
    # [NOTE]: not well tested
    unit_cnt = 0
    with orig_module.context, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):

            def instantivate_unit(name):
                nonlocal unit_cnt
                suffix = f"_{unit_cnt}"
                unit_cnt += 1
                # instantiate recursively
                unit: Unit = units[name]
                new_unit = unit.top.clone()
                new_unit.sym_name = StringAttr.get(new_unit.name.value + suffix)
                for block in new_unit.body:
                    for op in block:
                        if isinstance(op, func_d.CallOp):
                            assert op.callee.value in unit.works
                            work = unit.works[op.callee.value].clone()
                            work.sym_name = StringAttr.get(work.name.value + suffix)
                            op.callee = FlatSymbolRefAttr.get(work.sym_name.value)
                            for block in work.body:
                                replace(block.operations)

                return new_unit

            def replace(operations):
                for op in operations:
                    if isinstance(op, func_d.CallOp) and op.callee.value in units:
                        callee = instantivate_unit(op.callee.value)
                        op.callee = FlatSymbolRefAttr.get(callee.sym_name.value)

                    for region in op.regions:
                        for block in region.blocks:
                            replace(block.operations)

            top_unit: Unit = units[top_name]
            for work in top_unit.works.values():
                new_work = work.clone()
                for block in new_work.body:
                    replace(block.operations)

            top_unit.top.clone()
    print(module)

    return module
