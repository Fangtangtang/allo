# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-branches, too-many-nested-blocks, redefined-variable-type
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
import numpy as np

# =======================

import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.arith as aie_arith_d
import aie.dialects.func as aie_func_d
import aie.dialects.scf as aie_scf_d
import aie.dialects._memref_ops_gen as aie_memref_d

import aie.ir as aie_ir

# =======================

import allo._mlir._mlir_libs._mlir as allo_ir
import allo._mlir.dialects._memref_ops_gen as allo_memref_d

from ..._mlir.ir import InsertionPoint, MemRefType, IntegerType

from ..utils import format_str
from ..._mlir.dialects import func as allo_func_d
from ...memory import (
    DTensor,
    Offset4D,
    Size4D,
    coalesce_memory_access,
    format_memory_access,
)

from .utils import get_element_type, device_config_map, Argument, Stream, Config
from ..aie import map_kernels_to_device_mesh
from .mapping import (
    SwitchNode,
    OrderedDTensorTileGroup,
    PEInterface,
    DTensorTile,
    LiveDTensorTile,
    LiveDTensorTileGroup,
    DTensorTileGroup,
    ComputationGraph,
    FIFO,
    FIFOManager,
)


@dataclass(frozen=True)
class DMATensorTile:
    dtensor_tile_id: int  # dTensor may need to be further partitioned
    shim_id: int
    mem_id: int
    tensor_tile_labels: list
    offset: list
    size: list
    stride: list


def map_global_io(inputs, outputs) -> tuple[dict[str, list[DMATensorTile]], int, int]:
    """
    TODO: make use of the fifth mem tile
    TODO: The current mapping scheme requires the matrices to be completely partitioned without remaining elements (shape should be divided by tile num).

    Allocate (shim-tile, mem-tile) pairs for every DTensor that crosses the
    NPU boundary, while respecting the per-mem-tile ObjectFIFO limits.

    The algorithm tries to pack as much traffic as possible into the fewest
    number of memory tiles, only falling back to splitting (called "parts")
    when the requested number of FIFOs would exceed the quota per memory tile.

    Current constrains:
        - use 4 mem-shim tile pairs for io
        - each port is assigned to one dtensor tile

    Args:
        - inputs: A dictionary mapping function names (group name + id) to lists of objects as inputs.
        - outputs: A dictionary mapping function names (group name + id) to lists of objects as outputs.
    Return:
        - tile_map: dtensor name -> a list of dma tiles
        - mem_tile_num
        - shim_tile_num
    """
    MAX_MEM_TILES = 4  # Maximum number of memory tiles allowed

    @dataclass
    class Tile:
        send_number: int
        recv_number: int

    used_tiles: list[Tile] = []

    def assign_tile(send_need, recv_need) -> int:
        """
        Try to assign a memory tile satisfying the requirement.
        Return the tile index.
            -1 indicates no tile availability.
        """
        # 1. Attempt to use a new memory tile
        if (
            len(used_tiles) < MAX_MEM_TILES
            and send_need <= Config.MEM_MAX_SEND
            and recv_need <= Config.MEM_MAX_RECV
        ):
            used_tiles.append(Tile(send_need, recv_need))
            return len(used_tiles) - 1
        # 2. Otherwise, try to pack into an existing tile
        for i, _ in enumerate(used_tiles):
            if (
                used_tiles[i].send_number + send_need <= Config.MEM_MAX_SEND
                and used_tiles[i].recv_number + recv_need <= Config.MEM_MAX_RECV
            ):
                used_tiles[i].send_number += send_need
                used_tiles[i].recv_number += recv_need
                return i
        # 3. No tile fits
        return -1

    def map_dtensor_to_tile(dtensor: DTensor, is_input: bool):
        """
        Split a DTensor into Part instances so each Part fits on some memory tile with respect to FIFO limits.

        Currently, we focus on dtensor io using memory tiles.
        Shim tiles are assigned using a one-to-one mapping from memory tiles.

        DTensors are sent to or from compute cores.
        Since memory tile is used for transfer, we assume that `receive` implies one `send` and `send` implies one `receive`.
        """
        assert dtensor.access_pattern_set, "Access pattern is not set for dtensor"
        device_dims, size, stride = dtensor.shared_dims, dtensor.size, dtensor.stride
        tensor_tiles = sorted(
            list(dtensor.global_placement.keys())
        )  # 'R' can use one port yet multiple destinations
        send_need = len(tensor_tiles) if is_input else 1
        recv_need = 1 if is_input else len(tensor_tiles)
        mem_tile_id = assign_tile(send_need, recv_need)
        if mem_tile_id >= 0:
            return [
                DMATensorTile(
                    0,
                    mem_tile_id,
                    mem_tile_id,
                    tensor_tiles,
                    [0, 0, 0, 0],
                    size,
                    stride,
                )
            ]
        # We failed to transfer the whole tensor with one memory tile. Try using more.
        dma_tensor_tiles: list[DMATensorTile] = []
        # fixme: incomplete
        #   Currently, we may allow tensor tiles on a sharding dimension to be sent using different memory tiles
        if len(device_dims) <= 1:
            lose_factor, inc_factor = 1, 1
        elif len(device_dims) == 2:
            lose_factor = size[device_dims[0]]
            inc_factor = size[device_dims[1]]
        else:
            raise ValueError(f"Unsupported access pattern.")
        remaining = tensor_tiles[:]
        start_idx = 0
        while remaining:
            offset = [0, 0, 0, 0]
            chunk = remaining
            while chunk:
                send_need = len(chunk) if is_input else 1
                recv_need = 1 if is_input else len(chunk)
                mem_tile_id = assign_tile(send_need, recv_need)
                if mem_tile_id >= 0:
                    break
                chunk = chunk[: (len(chunk) - lose_factor)]  # Reduce size and retry
            if not chunk:
                raise RuntimeError(
                    "Failed to allocate (shim, memory) tile: per-tile FIFO limit "
                    "exceeded or no more available tiles."
                )
            offset[device_dims[0]] = start_idx // inc_factor
            size[device_dims[0]] = len(chunk) // inc_factor
            dma_tensor_tiles.append(
                DMATensorTile(
                    len(dma_tensor_tiles),
                    mem_tile_id,
                    mem_tile_id,
                    chunk,
                    offset,
                    size,
                    stride,
                )
            )
            remaining = remaining[len(chunk) :]
            start_idx += len(chunk)
        return dma_tensor_tiles

    tile_map: dict[str, list[DMATensorTile]] = defaultdict(list)

    for io_lst, is_input in ((inputs, True), (outputs, False)):
        for _, sub in io_lst.items():
            for dtensor in sub["_global"]:
                tile_map[dtensor.name].extend(
                    map_dtensor_to_tile(dtensor, is_input=is_input)
                )

    return tile_map, len(used_tiles), len(used_tiles)


class CodeGenerator:
    """
    CodeGenerator is responsible for transforming Allo functions and their associated
    DTensor-based input/output mappings into AIE (AI Engine) core-compatible IR.
    It manages stream transformations, memory operations, and integrates with the
    AIE dialect of MLIR.
    """

    def __init__(
        self,
        device_type: str,
        global_inputs: dict[int, DTensor],
        global_outputs: dict[int, DTensor],
        top_function: allo_func_d.FuncOp,
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
        streams: dict[str, Stream],
        virtual_computation_graph: ComputationGraph = None,
    ):
        self.device_type = device_type
        self.device_config = device_config_map[device_type]
        assert self.device_config is not None, "Unsupported device type"

        self.global_inputs: dict[int, DTensor] = global_inputs
        self.global_outputs: dict[int, DTensor] = global_outputs
        self.top_function = top_function
        self.core_func_args = core_func_args
        self.streams = streams
        self.virtual_computation_graph: ComputationGraph = virtual_computation_graph

        self.tile_map: dict[str, aie_d.TileOp] = {}
        self.fifo_map: dict[str, aie_d.object_fifo] = {}
        # function name (with id) -> a map from DTensor to fifo name
        self.compute_core_io: dict[str : dict[DTensor, str]] = {}
        # function name (with id) -> a map from DTensor to (fifo name, transfer idx, total tensfer number)
        self.compute_core_io_experimental: dict[
            str : dict[DTensor, tuple[str, int]]
        ] = {}
        self.external_functions: str = ""

        # ------------------------------------------------------------
        # Experimental
        # ------------------------------------------------------------
        self.used_mem_tiles: list[SwitchNode] = None
        self.used_shim_tiles: list[SwitchNode] = None
        self.global_io_dma: dict[str, list[CodeGenerator.GlobalIODMA]] = None
        self.function_port_map: dict[str, dict[DTensor, SwitchNode.Port]] = defaultdict(
            lambda: defaultdict(SwitchNode.Port)
        )
        self.exp_tile_map: dict[str, aie_d.TileOp] = {}
        self.exp_fifo_map: dict[str, aie_d.object_fifo] = {}

        self.fifo_manager: FIFOManager = FIFOManager()
        self.experimental_fifo_manager: FIFOManager = FIFOManager()

        self.exp_aie_module = None  # The top-level AIE IR module
        self.exp_global_ip: aie_ir.InsertionPoint = (
            None  # mark the inserting point for buffers
        )
        # ------------------------------------------------------------

        self.aie_module = None  # The top-level AIE IR module
        self.global_ip: aie_ir.InsertionPoint = (
            None  # mark the inserting point for buffers
        )

    def preporocess_dumped_core_func(
        self,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
    ) -> str:
        """
        Preprocess the core function in allo MLIR.

        Args:
            - original_func (FuncOp): The function in allo MLIR to transform.
            - func_args (dict): Maps function argument indices to (Argument, is_output) pairs.

        Returns:
            - str: A string representation of the rewritten function with allo.stream ops replaced.
        """
        # replace pipe with memref operations
        with original_func.context, allo_ir.ir.Location.unknown():
            func_inputs = original_func.type.inputs
            new_func_inputs = []
            for idx in range(len(func_inputs)):
                if idx in func_args and func_args[idx][0].stream is not None:
                    new_func_inputs.append(func_args[idx][0].stream.allo_element_type)
                    func_inputs[idx] = func_args[idx][0].stream.allo_element_type
                elif idx in func_args and func_args[idx][0].dtensor is not None:
                    new_func_inputs.append(func_inputs[idx])
                else:
                    # fixme: this is a fake placeholder, we'd better remove the useless argument, but doing so leads to crash
                    #           "Cannot destroy a value that still has uses!"
                    new_func_inputs.append(
                        MemRefType.get([], IntegerType.get_signless(8))
                    )

            print(new_func_inputs)
            func_type = allo_func_d.FunctionType.get(
                new_func_inputs,
                original_func.type.results,
            )
            new_function = allo_func_d.FuncOp(
                original_func.name.value,
                func_type,
                ip=InsertionPoint(original_func),
            )
            entry_block = new_function.add_entry_block()
            for old, new in zip(original_func.arguments, new_function.arguments):
                old.replace_all_uses_with(new)

            with InsertionPoint(entry_block):
                for func_block in original_func.body:
                    for op in func_block.operations:
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
            original_func.erase()
            for idx, arg_info in func_args.items():
                if arg_info[0].stream is not None:
                    argument = new_function.arguments[idx]
                    for use_ in argument.uses:
                        op = use_.owner
                        if op.name == "allo.stream_put":
                            operands = op.operands
                            # store/copy
                            if arg_info[0].stream.is_tensor:
                                new_op = allo_memref_d.CopyOp(
                                    operands[1], operands[0], ip=InsertionPoint(op)
                                )
                            else:
                                new_op = allo_memref_d.StoreOp(
                                    operands[1], operands[0], [], ip=InsertionPoint(op)
                                )
                        elif op.name == "allo.stream_get":
                            # load/alloc
                            if arg_info[0].stream.is_tensor:
                                # replace use with alloc
                                new_op = allo_memref_d.AllocOp(
                                    arg_info[0].stream.allo_element_type,
                                    [],
                                    [],
                                    ip=InsertionPoint(op),
                                )
                                # use copy to track
                                allo_memref_d.CopyOp(
                                    op.operands[0], new_op.memref, ip=InsertionPoint(op)
                                )
                            else:
                                new_op = allo_memref_d.LoadOp(
                                    argument, [], ip=InsertionPoint(op)
                                )
                        else:
                            continue
                        # replace use
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                        op.erase()

        # declare external kernel function before use
        func_str = self.external_functions + "\n" + str(new_function)
        return func_str

    def build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
        is_experimental: bool = False,
    ):
        """
        Generate the computation logic for the fake 'while(1)' loop body for an AIE compute core, transforming high-level Allo ops
        into AIE MLIR.

        Args:
            - func_core (aie_d.Core): The target compute core to insert into.
            - original_func (FuncOp): The Allo function to compile.
            - func_args (dict): Maps argument indices to (Argument, is_output) tuples.
        """
        func_string = self.preporocess_dumped_core_func(original_func, func_args)
        original_module = aie_ir.Module.parse(func_string)
        parsed_function: aie_func_d.FuncOp = None
        for func in original_module.body.operations:
            if isinstance(func, aie_func_d.FuncOp):
                if not (
                    "sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    if parsed_function is None:
                        parsed_function = func
                    else:
                        raise ValueError("Too many core functions. Fail to resolve.")
        assert not parsed_function is None
        func_body = func_core.regions[0]
        entry_block = aie_ir.Block.create_at_start(func_body)
        with aie_ir.InsertionPoint(entry_block):
            index_type = aie_ir.IndexType.get()
            # compute core wrapper: fake while(1)
            c0 = aie_arith_d.ConstantOp(value=0, result=index_type)
            c1 = aie_arith_d.ConstantOp(value=1, result=index_type)
            cmax = aie_arith_d.ConstantOp(value=9223372036854775807, result=index_type)
            # scf.for %arg0 = %c0 to %cmax step %c1
            loop = aie_scf_d.ForOp(lower_bound=c0, upper_bound=cmax, step=c1)
            with aie_ir.InsertionPoint(loop.body):
                # insert operations to get 'function parameter', acquire and subview
                # fixme: arguments may reuse fifo (ordering and copy-release-acquire)
                # fixme: what about the output??
                compute_core_io = (
                    self.compute_core_io_experimental
                    if is_experimental
                    else self.compute_core_io
                )
                io_map = (
                    compute_core_io[parsed_function.name.value]
                    if parsed_function.name.value in compute_core_io
                    else {}
                )
                argument_info: dict[str, list[tuple]] = defaultdict(list)
                if is_experimental:
                    for i, argument in enumerate(parsed_function.arguments):
                        if not i in func_args:
                            continue
                        arg_info: tuple[Argument, bool] = func_args[i]
                        if arg_info[0].dtensor is not None:
                            fifo_name, transfer_idx = io_map[arg_info[0].dtensor]
                            argument_info[fifo_name].append(
                                (transfer_idx, argument, arg_info[1])
                            )
                    for key in argument_info:
                        argument_sharing_fifo = argument_info[key]
                        assert len(argument_sharing_fifo) >= 1
                        argument_info[key].sort(key=lambda x: x[0])
                        idx = 0
                        while idx < len(argument_sharing_fifo) - 2:
                            argument = argument_sharing_fifo[idx][1]
                            acquired = self.fifo_map[key].acquire(
                                1 if argument_sharing_fifo[idx][2] else 0, 1
                            )
                            assert argument_sharing_fifo[idx][
                                2
                            ], "At most one global output is supported"
                            alloc_op = aie_memref_d.AllocOp(
                                acquired.type,
                                [],
                                [],
                            )
                            aie_memref_d.CopyOp(
                                acquired,
                                alloc_op.memref,
                            )
                            argument.replace_all_uses_with(alloc_op.memref)
                            idx += 1
                        argument = argument_sharing_fifo[idx][1]
                        acquired = self.fifo_map[key].acquire(
                            1 if argument_sharing_fifo[idx][2] else 0, 1
                        )
                        argument.replace_all_uses_with(acquired)

                for i, argument in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if arg_info[0].dtensor is not None:
                        if not is_experimental:
                            acquired = self.fifo_map[
                                io_map[arg_info[0].dtensor]
                            ].acquire(1 if arg_info[1] else 0, 1)
                            argument.replace_all_uses_with(acquired)
                    else:
                        stream: Stream = arg_info[0].stream
                        fifo = self.fifo_map[stream.name]
                        for use_ in argument.uses:
                            op = use_.owner
                            with aie_ir.InsertionPoint(op.operation):
                                if op.name == "memref.store" or (
                                    op.name == "memref.copy"
                                    and argument == op.operands[1]
                                ):  # allo.stream_put
                                    acquired = fifo.acquire(0, 1)
                                    op.operands[1] = acquired
                                    new_op = op.clone()  # no use, no need to replace
                                    fifo.release(0, 1)
                                    op.erase()
                                elif (
                                    op.name == "memref.load"
                                ):  # allo.stream_get, non-tensor
                                    acquired = fifo.acquire(1, 1)
                                    op.operands[0] = acquired
                                    new_op = op.clone()
                                    for old, new in zip(op.results, new_op.results):
                                        old.replace_all_uses_with(new)
                                    fifo.release(1, 1)
                                    op.erase()
                                elif (
                                    op.name == "memref.copy"
                                ):  # allo.stream_get, tensor
                                    acquired = fifo.acquire(1, 1)
                                    op.operands[0] = acquired
                                    new_op = op.clone()
                                    fifo.release(1, 1)
                                    op.erase()

                for parsed_func_block in parsed_function.body:
                    for op in parsed_func_block.operations:
                        if op.name == "func.return":
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)

                # replace alloc with buffer
                alloc_ops = []

                def collect_allocs(op):
                    if op.name == "memref.alloc":
                        alloc_ops.append(op.operation)
                        return
                    for region in op.regions:
                        for block in region.blocks:
                            for inner_op in block.operations:
                                collect_allocs(inner_op)

                collect_allocs(loop)
                for alloc_op in alloc_ops:
                    buffer_op = aie_d.BufferOp(
                        buffer=alloc_op.results[0].type,
                        tile=func_core.tile,
                        ip=self.global_ip,
                    )
                    for old, new in zip(alloc_op.results, buffer_op.results):
                        old.replace_all_uses_with(new)
                    alloc_op.erase()

                # release
                if is_experimental:
                    for key in argument_info:
                        self.fifo_map[key].release(
                            1 if argument_info[key][0][2] else 0, 1
                        )
                for i, _ in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if not arg_info[0].dtensor is None:
                        if not is_experimental:
                            self.fifo_map[io_map[arg_info[0].dtensor]].release(
                                1 if arg_info[1] else 0, 1
                            )

                aie_scf_d.YieldOp([])
            aie_d.EndOp()

    def exp_build_core_function(
        self,
        func_core: aie_d.Core,
        original_func: allo_func_d.FuncOp,
        func_args: dict[int, tuple[Argument, bool]],
        interfaces: dict[int, FIFO],
    ):
        """
        Generate the computation logic for the fake 'while(1)' loop body for an AIE compute core, transforming high-level Allo ops
        into AIE MLIR.

        fixme: current constrain
            - all the argument using the same port has no overlapped liveness range
            - the usage order is aligned to the data transfer order

        Args:
            - func_core (aie_d.Core): The target compute core to insert into.
            - original_func (FuncOp): The Allo function to compile.
            - func_args (dict): Maps argument indices to (Argument, is_output) tuples.
        """
        func_string = self.preporocess_dumped_core_func(original_func, func_args)
        original_module = aie_ir.Module.parse(func_string)
        parsed_function: aie_func_d.FuncOp = None
        for func in original_module.body.operations:
            if isinstance(func, aie_func_d.FuncOp):
                if not (
                    "sym_visibility" in func.attributes
                    and func.attributes["sym_visibility"].value == "private"
                ):
                    if parsed_function is None:
                        parsed_function = func
                    else:
                        raise ValueError("Too many core functions. Fail to resolve.")
        assert not parsed_function is None
        print("\n#------------------")
        print(parsed_function)
        print("#------------------\n")
        func_body = func_core.regions[0]
        entry_block = aie_ir.Block.create_at_start(func_body)
        with aie_ir.InsertionPoint(entry_block):
            index_type = aie_ir.IndexType.get()
            # compute core wrapper: fake while(1)
            c0 = aie_arith_d.ConstantOp(value=0, result=index_type)
            c1 = aie_arith_d.ConstantOp(value=1, result=index_type)
            cmax = aie_arith_d.ConstantOp(value=9223372036854775807, result=index_type)
            # scf.for %arg0 = %c0 to %cmax step %c1
            loop = aie_scf_d.ForOp(lower_bound=c0, upper_bound=cmax, step=c1)
            reused_fifo_name: dict[str, bool] = {}
            for i, argument in enumerate(parsed_function.arguments):
                if not i in func_args:
                    continue
                arg_info: tuple[Argument, bool] = func_args[i]
                if arg_info[0].dtensor is not None:
                    first_use = next(iter(argument.uses), None)
                    if first_use is not None:
                        first_use_op = first_use.owner
                        # no branch
                        while first_use_op.parent.name != "func.func":
                            first_use_op = first_use_op.parent
                        fifo = self.exp_fifo_map[interfaces[i].name]
                        is_input = arg_info[0].dtensor in self.global_inputs
                        with aie_ir.InsertionPoint(first_use_op.operation):
                            if interfaces[i].name in reused_fifo_name:
                                fifo.release(1 if is_input else 0, 1)
                            else:
                                reused_fifo_name[interfaces[i].name] = is_input
                            acquired = fifo.acquire(1 if is_input else 0, 1)
                            # incorrect
                            argument.replace_all_uses_with(acquired)
                    pass
                else:
                    stream: Stream = arg_info[0].stream
                    fifo = self.fifo_map[stream.name]
                    for use_ in argument.uses:
                        op = use_.owner
                        with aie_ir.InsertionPoint(op.operation):
                            if op.name == "memref.store" or (
                                op.name == "memref.copy" and argument == op.operands[1]
                            ):  # allo.stream_put
                                acquired = fifo.acquire(0, 1)
                                op.operands[1] = acquired
                                new_op = op.clone()  # no use, no need to replace
                                fifo.release(0, 1)
                                op.erase()
                            elif (
                                op.name == "memref.load"
                            ):  # allo.stream_get, non-tensor
                                acquired = fifo.acquire(1, 1)
                                op.operands[0] = acquired
                                new_op = op.clone()
                                for old, new in zip(op.results, new_op.results):
                                    old.replace_all_uses_with(new)
                                fifo.release(1, 1)
                                op.erase()
                            elif op.name == "memref.copy":  # allo.stream_get, tensor
                                acquired = fifo.acquire(1, 1)
                                op.operands[0] = acquired
                                new_op = op.clone()
                                fifo.release(1, 1)
                                op.erase()

            with aie_ir.InsertionPoint(loop.body):
                for parsed_func_block in parsed_function.body:
                    for op in parsed_func_block.operations:
                        if op.name == "func.return":
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                # replace alloc with buffer
                alloc_ops = []

                def collect_allocs(op):
                    if op.name == "memref.alloc":
                        alloc_ops.append(op.operation)
                        return
                    for region in op.regions:
                        for block in region.blocks:
                            for inner_op in block.operations:
                                collect_allocs(inner_op)

                collect_allocs(loop)
                for alloc_op in alloc_ops:
                    buffer_op = aie_d.BufferOp(
                        buffer=alloc_op.results[0].type,
                        tile=func_core.tile,
                        ip=self.global_ip,
                    )
                    for old, new in zip(alloc_op.results, buffer_op.results):
                        old.replace_all_uses_with(new)
                    alloc_op.erase()

                for fifo_name, is_input in reused_fifo_name.items():
                    self.exp_fifo_map[fifo_name].release(1 if is_input else 0, 1)

                aie_scf_d.YieldOp([])
            aie_d.EndOp()

    # ------------------------------------------------------------
    # Data Transfer
    # ------------------------------------------------------------

    def analyze_global_io(self) -> tuple[
        dict[int, int],
        dict[int, OrderedDTensorTileGroup],
    ]:
        """
        return:
            - global_io_ordering: dtensor id -> ordering tag
            - global_tile_to_func: dtensor id -> related tile group
        """
        # global inputs/outputs
        global_tensors = self.virtual_computation_graph.get_node_global_io()
        # manage the order to avoid deadlocks
        dependencies = self.virtual_computation_graph.get_node_dependencies()
        node_order_tag: dict[str, int] = {}
        tag = 0
        while len(dependencies.items()) > 0:
            tagged_nodes = []
            for node, deps in dependencies.items():
                if len(deps) == 0:
                    node_order_tag[node] = tag
                    tagged_nodes.append(node)
            for node in tagged_nodes:
                del dependencies[node]
                for _, deps in dependencies.items():
                    if node in deps:
                        deps.remove(node)
            tag += 1

        if os.getenv("VERBOSE") == "1":
            print("\n<<<<<<< global_tensors >>>>>>>>")
            for func_name, global_io in global_tensors.items():
                print(func_name)
                for key, value in global_io.items():
                    print(
                        "\t", key, ": [", ", ".join([str(tile) for tile in value]), "]"
                    )
            print()
            print(node_order_tag)

        global_io_ordering: dict[int, int] = {}
        all_keys = self.global_inputs.keys() | self.global_outputs.keys()
        global_tile_to_func: dict[int, OrderedDTensorTileGroup] = {
            i: OrderedDTensorTileGroup() for i in all_keys
        }
        for func_name, io_info in global_tensors.items():
            outer_tag = node_order_tag[func_name]
            for arg_idx, tiles in io_info.items():
                for i, tile_ in enumerate(tiles):
                    global_tile_to_func[tile_.dtensor_id].add_tensor_tile(
                        tile_, f"{outer_tag}-{i}", func_name, arg_idx
                    )
                    global_io_ordering[tile_.dtensor_id] = (
                        outer_tag
                        if tile_.dtensor_id not in global_io_ordering
                        else min(outer_tag, global_io_ordering[tile_.dtensor_id])
                    )

        if os.getenv("VERBOSE") == "1":
            print("\n<<<<<<< global_tile_to_func >>>>>>>>")
            for i in global_tile_to_func.keys():
                global_tile_to_func[i].print()
            print("\n<<<<<<< global_io_ordering >>>>>>>>")
            print(global_io_ordering)

        return global_io_ordering, global_tile_to_func

    @dataclass(frozen=True)
    class GlobalIODMA:
        dtensor: DTensor
        port: SwitchNode.Port
        offset: list[int]
        size: list[int]
        stride: list[int]
        is_input: bool

    @staticmethod
    def parse_tag2(tag: str) -> tuple[int, int]:
        outer_tag, inner_tag = tag.split("-")
        return int(outer_tag), int(inner_tag)

    @staticmethod
    def parse_tag3(tag: str) -> tuple[int, int, int]:
        outer_tag, inner_tag, ext_tag = tag.split("-")
        return int(outer_tag), int(inner_tag), int(ext_tag)

    def map_global_io_to_physical_tiles(
        self,
    ) -> tuple[
        list[SwitchNode],
        list[SwitchNode],
        dict[str, list["CodeGenerator.GlobalIODMA"]],
    ]:
        """
        Map the global io to physical tiles. (memory/shim tiles in AIE)
        """
        # ------------------------------------------------------------
        MAX_MEM_TILES = self.device_config["mem_tile_num"]
        MAX_SHIM_TILES = self.device_config["shim_tile_num"]

        self.used_mem_tiles = []
        self.used_shim_tiles = []
        self.global_io_dma = defaultdict(list)
        # ------------------------------------------------------------

        def assign_mem_tile(
            dtype: str,
            connected_nodes: list[list[str]],
            is_input: bool,
            coalesced_size: Size4D,
            tile_size: Size4D,
            tile_shape: list[int],
        ) -> tuple[SwitchNode, int, list[int]]:
            """
            fixme: maybe too aie-specific?
            Assign a memory tile to the given dtensor tiles.
            If no memory tile is available, return None.
            Else, return the assigned memory tile, the port id to shim, and the port ids to compute.
            """
            send_need = len(connected_nodes) if is_input else 1
            recv_need = 1 if is_input else sum(len(group) for group in connected_nodes)
            send_shape: list[int] = tile_shape if is_input else coalesced_size.to_list()
            recv_shape: list[int] = coalesced_size.to_list() if is_input else tile_shape
            tile_total_size = tile_size.get_total_size()
            if os.getenv("VERBOSE") == "1":
                print(f"send_need: {send_need}, recv_need: {recv_need}")
            assigned_mem_tile = None
            # Attempt to use a new memory tile
            if (
                len(self.used_mem_tiles) < MAX_MEM_TILES
                and send_need <= Config.MEM_MAX_SEND
                and recv_need <= Config.MEM_MAX_RECV
            ):
                assigned_mem_tile = SwitchNode(
                    name=f"{len(self.used_mem_tiles)}_mem_tile",
                    send_port_num=Config.MEM_MAX_SEND,
                    recv_port_num=Config.MEM_MAX_RECV,
                )
                self.used_mem_tiles.append(assigned_mem_tile)
            else:
                # Attempt to use an existing memory tile
                for mem_tile in self.used_mem_tiles:
                    if (
                        len(mem_tile.send_ports) + send_need <= Config.MEM_MAX_SEND
                        and len(mem_tile.recv_ports) + recv_need <= Config.MEM_MAX_RECV
                    ):
                        assigned_mem_tile = mem_tile
                        break
            # Use new ports
            if assigned_mem_tile is not None:
                send_ports, recv_ports = [], []
                for i in range(send_need):
                    port = SwitchNode.Port(
                        id=len(assigned_mem_tile.send_ports),
                        data_shape=send_shape,
                        dtype=dtype,
                        connected_nodes=connected_nodes[i] if is_input else [],
                    )
                    assigned_mem_tile.send_ports.append(port)
                    send_ports.append(port.id)
                if is_input:
                    for i in range(recv_need):
                        port = SwitchNode.Port(
                            id=len(assigned_mem_tile.recv_ports),
                            data_shape=recv_shape,
                            dtype=dtype,
                            connected_nodes=[],
                        )
                        assigned_mem_tile.recv_ports.append(port)
                        recv_ports.append(port.id)
                else:
                    for group in connected_nodes:
                        for node in group:
                            port = SwitchNode.Port(
                                id=len(assigned_mem_tile.recv_ports),
                                data_shape=recv_shape,
                                dtype=dtype,
                                connected_nodes=[node],
                            )
                            assigned_mem_tile.recv_ports.append(port)
                            recv_ports.append(port.id)
                assigned_mem_tile.intra_connect.append(
                    SwitchNode.IntraConnect(
                        send_ports,
                        recv_ports,
                        list(
                            range(
                                0,
                                max(send_need, recv_need) * tile_total_size,
                                tile_total_size,
                            )
                        ),
                    )
                )
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_mem_tile: ", end="")
                    assigned_mem_tile.print()
                return (
                    assigned_mem_tile,
                    recv_ports[0] if is_input else send_ports[0],
                    send_ports if is_input else recv_ports,
                )
            # TODO: port reuse
            return None, -1, []

        def assign_shim_tile(
            mem_tile: SwitchNode,
            mem_port: SwitchNode.Port,
            is_input: bool,
        ) -> tuple[SwitchNode, int]:
            assigned_shim_tile = None
            # Attempt to use a new shim tile
            if len(self.used_shim_tiles) < MAX_SHIM_TILES:
                assigned_shim_tile = SwitchNode(
                    name=f"{len(self.used_shim_tiles)}_shim_tile",
                    send_port_num=Config.SHIM_MAX_SEND,
                    recv_port_num=Config.SHIM_MAX_RECV,
                )
                self.used_shim_tiles.append(assigned_shim_tile)
            else:
                for shim_tile in self.used_shim_tiles:
                    if (
                        len(shim_tile.send_ports) + 1 <= Config.SHIM_MAX_SEND
                        and len(shim_tile.recv_ports) + 1 <= Config.SHIM_MAX_RECV
                    ):
                        assigned_shim_tile = shim_tile
                        break
            # Use new ports
            if assigned_shim_tile is not None:
                connected_mem = [mem_tile.name]
                send_port = SwitchNode.Port(
                    id=len(assigned_shim_tile.send_ports),
                    data_shape=mem_port.data_shape,
                    dtype=mem_port.dtype,
                    connected_nodes=connected_mem if is_input else [],
                )
                assigned_shim_tile.send_ports.append(send_port)
                recv_port = SwitchNode.Port(
                    id=len(assigned_shim_tile.recv_ports),
                    data_shape=mem_port.data_shape,
                    dtype=mem_port.dtype,
                    connected_nodes=[] if is_input else connected_mem,
                )
                assigned_shim_tile.recv_ports.append(recv_port)
                mem_port.connected_nodes.append(assigned_shim_tile.name)
                assigned_shim_tile.intra_connect.append(
                    SwitchNode.IntraConnect(
                        send_port_ids=[send_port.id],
                        recv_port_ids=[recv_port.id],
                        offsets=[0],
                    )
                )
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_shim_tile: ", end="")
                    assigned_shim_tile.print()
                return assigned_shim_tile, recv_port.id if is_input else send_port.id
                # TODO: port reuse
            return None, -1

        def map_dtensor_to_physical_tiles(
            dtensor: DTensor,
            ordered_tile_group: OrderedDTensorTileGroup,
            is_input: bool,
        ):
            def partition(size: Size4D) -> Size4D:
                """
                Partition the dma task into multiple sub-tasks.
                """
                # find the first none-1 dim
                for dim in range(4):
                    if size.get_dim_size(dim) > 1:
                        break
                if dim >= 3:
                    raise ValueError("Fail to partition")
                size_part = size.copy()
                partition_size = size.get_dim_size(dim) - 1
                size_part.set_dim_size(dim, partition_size)
                return size_part

            def register_function_param_port(port: SwitchNode.Port):
                for node in port.connected_nodes:
                    self.function_port_map[node][dtensor] = port

            tile_dtype = dtensor.dtype
            tile_shape = list(dtensor.size)
            for i in dtensor.shared_dims:
                tile_shape[i] = 1
            tile_size = Size4D.from_list(tile_shape)
            # Tags sorted in lexicographic order are used to preserve the data transfer sequence.
            # tiles in DMATileGroup with the same tage can be sent in parallel.
            sorted_tags = sorted(
                list(ordered_tile_group.dtensor_tile_groups.keys()),
                key=CodeGenerator.parse_tag2,
            )
            idx = 0
            while idx < len(sorted_tags):
                tag = sorted_tags[idx]
                update = 0
                offset_map: dict[Offset4D, list[str]] = {}
                # fixme: this is an ugly and problematic hack. We need more elegant and robust way to handle this.
                while len(
                    offset_map
                ) < Config.IO_TILE_LOSE_FACTOR and idx + update < len(sorted_tags):
                    dma_tile_group = ordered_tile_group.dtensor_tile_groups[
                        sorted_tags[idx + update]
                    ]
                    for dma_tile in dma_tile_group.dtensor_tile_to_pe_interfaces.keys():
                        if (
                            dtensor.offset_map[dma_tile.tensor_tile_label]
                            not in offset_map
                        ):
                            offset_map[
                                dtensor.offset_map[dma_tile.tensor_tile_label]
                            ] = []
                        for (
                            pe_interface
                        ) in dma_tile_group.dtensor_tile_to_pe_interfaces[dma_tile]:
                            # fixme: the argument idx should guide the mapping
                            offset_map[
                                dtensor.offset_map[dma_tile.tensor_tile_label]
                            ].append(pe_interface.pe)

                    update += 1
                access, coalesce_info, connected_nodes = coalesce_memory_access(
                    offset_map
                )
                coalesced_access, fallback_flag = format_memory_access(
                    access, coalesce_info, connected_nodes
                )

                if os.getenv("VERBOSE") == "1":
                    print()
                    print("tag:", tag, "update:", update)
                    print(offset_map)
                    print("access:", coalesced_access)
                offset_id = 0
                for offset, mem_access in coalesced_access.items():
                    connected_nodes: list[list[str]] = mem_access.connected_nodes
                    size = mem_access.transfer_size
                    multiplier: Size4D = Size4D.divide(
                        mem_access.total_size, mem_access.transfer_size
                    )
                    while size.get_total_size() != 0:
                        coalesced_size = Size4D.coalesce(size, tile_size)
                        assigned_mem_tile, port_id, ports_to_compute = assign_mem_tile(
                            tile_dtype,
                            connected_nodes,
                            is_input,
                            coalesced_size,
                            tile_size,
                            tile_shape=dtensor.type_as_param,
                        )
                        if assigned_mem_tile is not None:
                            port_to_shim = (
                                assigned_mem_tile.recv_ports[port_id]
                                if is_input
                                else assigned_mem_tile.send_ports[port_id]
                            )
                            extened_tag = f"{tag}-{len(port_to_shim.queue)}"
                            port_to_shim.queue.append(dtensor)
                            for port in ports_to_compute:
                                port_to_compute = (
                                    assigned_mem_tile.send_ports[port]
                                    if is_input
                                    else assigned_mem_tile.recv_ports[port]
                                )
                                port_to_compute.queue.append(dtensor)
                                register_function_param_port(port_to_compute)
                            assigned_shim_tile, shim_port_id = assign_shim_tile(
                                assigned_mem_tile,
                                port_to_shim,
                                is_input,
                            )
                            if assigned_shim_tile is None:
                                raise ValueError("Fail to assign shim tile")
                            port = (
                                assigned_shim_tile.send_ports[shim_port_id]
                                if is_input
                                else assigned_shim_tile.recv_ports[shim_port_id]
                            )
                            port.queue.append(dtensor)
                            self.global_io_dma[extened_tag].append(
                                CodeGenerator.GlobalIODMA(
                                    dtensor=dtensor,
                                    port=port,
                                    offset=mem_access.offset_info[offset_id].to_list(),
                                    size=Size4D.coalesce(
                                        Size4D.multiply(multiplier, size), tile_size
                                    ).to_list(),
                                    stride=dtensor.stride,
                                    is_input=is_input,
                                )
                            )
                            break
                        size_cp = size.copy()
                        # keep partitioning until success
                        while True:
                            partitioned_size = partition(size_cp)
                            coalesced_size = Size4D.coalesce(
                                partitioned_size, tile_size
                            )
                            partitioned_connected_nodes = connected_nodes[
                                : partitioned_size.get_total_size()
                            ]
                            assigned_mem_tile, port_id, ports_to_compute = (
                                assign_mem_tile(
                                    tile_dtype,
                                    partitioned_connected_nodes,
                                    is_input,
                                    coalesced_size,
                                    tile_size,
                                    tile_shape=dtensor.type_as_param,
                                )
                            )
                            if assigned_mem_tile is not None:
                                port_to_shim = (
                                    assigned_mem_tile.recv_ports[port_id]
                                    if is_input
                                    else assigned_mem_tile.send_ports[port_id]
                                )
                                extened_tag = f"{tag}-{len(port_to_shim.queue)}"
                                port_to_shim.queue.append(dtensor)
                                for port in ports_to_compute:
                                    port_to_compute = (
                                        assigned_mem_tile.send_ports[port]
                                        if is_input
                                        else assigned_mem_tile.recv_ports[port]
                                    )
                                    port_to_compute.queue.append(dtensor)
                                    register_function_param_port(port_to_compute)
                                assigned_shim_tile, shim_port_id = assign_shim_tile(
                                    assigned_mem_tile,
                                    port_to_shim,
                                    is_input,
                                )
                                if assigned_shim_tile is None:
                                    raise ValueError("Fail to assign shim tile")
                                port = (
                                    assigned_shim_tile.send_ports[shim_port_id]
                                    if is_input
                                    else assigned_shim_tile.recv_ports[shim_port_id]
                                )
                                port.queue.append(dtensor)
                                self.global_io_dma[extened_tag].append(
                                    CodeGenerator.GlobalIODMA(
                                        dtensor=dtensor,
                                        port=port,
                                        offset=mem_access.offset_info[
                                            offset_id
                                        ].to_list(),
                                        size=Size4D.coalesce(
                                            Size4D.multiply(
                                                multiplier, partitioned_size
                                            ),
                                            tile_size,
                                        ).to_list(),
                                        stride=dtensor.stride,
                                        is_input=is_input,
                                    )
                                )
                                break
                            size_cp = partitioned_size
                        size = Size4D.subtract(size, partitioned_size)
                        inc = partitioned_size.get_total_size()
                        connected_nodes = connected_nodes[inc:]
                        offset_id += inc
                idx += update

        global_io_ordering, global_tile_to_func = self.analyze_global_io()
        # fixme: what if the Tensor is send multiple times?
        sorted_keys = sorted(global_io_ordering, key=lambda k: global_io_ordering[k])
        for key in sorted_keys:
            if key in self.global_inputs:
                map_dtensor_to_physical_tiles(
                    self.global_inputs[key],
                    global_tile_to_func[key],
                    is_input=True,
                )
            elif key in self.global_outputs:
                map_dtensor_to_physical_tiles(
                    self.global_outputs[key],
                    global_tile_to_func[key],
                    is_input=False,
                )
            else:
                raise ValueError("Run into an unreachable point.")

        if os.getenv("VERBOSE") == "1":
            print("\n\n########################################################")
            print("used_mem_tiles:")
            for mem_tile in self.used_mem_tiles:
                mem_tile.print()
            print("\nused_shim_tiles:")
            for shim_tile in self.used_shim_tiles:
                shim_tile.print()
            print("\nglobal_io_dma:")
            for tag, dma_list in self.global_io_dma.items():
                print(f"#: {tag}")
                for dma in dma_list:
                    print(f"    {dma}")
            print("########################################################\n\n")

    def map_data_transfer(self) -> dict[str, dict[int, FIFO]]:

        def partition(size: Size4D) -> Size4D:
            """
            Partition the dma task into multiple sub-tasks.
            """
            # find the first none-1 dim
            for dim in range(4):
                if size.get_dim_size(dim) > 1:
                    break
            if dim >= 3:
                raise ValueError("Fail to partition")
            size_part = size.copy()
            partition_size = size.get_dim_size(dim) - 1
            size_part.set_dim_size(dim, partition_size)
            return size_part

        # ------------------------------------------------------------
        MAX_MEM_TILES = self.device_config["mem_tile_num"]
        MAX_SHIM_TILES = self.device_config["shim_tile_num"]

        self.exp_used_mem_tiles: list[SwitchNode] = []
        self.exp_used_shim_tiles: list[SwitchNode] = []
        self.exp_global_io_dma = defaultdict(list)
        # ------------------------------------------------------------

        dependencies = self.virtual_computation_graph.get_node_dependencies()
        ordered_nodes: list[str] = []  # topological order
        node_order_tag: dict[str, int] = {}
        tag = 0
        while len(dependencies.items()) > 0:
            tagged_nodes = []
            for node, deps in dependencies.items():
                if len(deps) == 0:
                    node_order_tag[node] = tag
                    tagged_nodes.append(node)
            ordered_nodes.extend(tagged_nodes)
            for node in tagged_nodes:
                del dependencies[node]
                for _, deps in dependencies.items():
                    if node in deps:
                        deps.remove(node)
            tag += 1

        # func name -> (arg idx -> dtensor tiles using that arg)
        global_tensors: dict[str, dict[int, LiveDTensorTileGroup]] = (
            self.virtual_computation_graph.get_global_io()
        )
        global_dtensor: dict[int, DTensor] = dict(self.global_inputs)
        global_dtensor.update(self.global_outputs)
        global_tile_to_func: dict[int, DTensorTileGroup] = {
            i: DTensorTileGroup("") for i in global_dtensor.keys()
        }
        for func_name, io_info in global_tensors.items():
            for arg_idx, live_dtensor_tiles in io_info.items():
                for tiles in live_dtensor_tiles.dtensor_groups.values():
                    for tile_ in tiles:
                        global_tile_to_func[tile_.tile.dtensor_id].add_tensor_tile(
                            tile_.tile, func_name, arg_idx
                        )

        class MulticastInterface:
            """
            MulticastInterface use the same port from source tile
            """

            def __init__(self, interface: PEInterface):
                self.sample_interface: PEInterface = interface
                self.interface_list: set[PEInterface] = {interface}

            def _equal_data_transfer(self, other: "MulticastInterface") -> bool:
                sample_global_tensors: LiveDTensorTileGroup = global_tensors[
                    self.sample_interface.pe
                ][self.sample_interface.interface_idx]
                other_global_tensor: LiveDTensorTileGroup = global_tensors[
                    other.sample_interface.pe
                ][other.sample_interface.interface_idx]
                if len(sample_global_tensors.dtensor_groups) == len(
                    other_global_tensor.dtensor_groups
                ):
                    sample_value = next(
                        iter(sample_global_tensors.dtensor_groups.values())
                    )
                    other_value = next(
                        iter(other_global_tensor.dtensor_groups.values())
                    )
                    return sample_value == other_value
                else:
                    return False

            def _contiguous_data_transfer(
                self, other: "MulticastInterface", current_size: Size4D
            ) -> Size4D:
                for interface in self.interface_list:
                    if interface in other.interface_list:
                        # TODO: can be relaxed
                        return None
                sample_global_tensors: LiveDTensorTileGroup = global_tensors[
                    self.sample_interface.pe
                ][self.sample_interface.interface_idx]
                other_global_tensor: LiveDTensorTileGroup = global_tensors[
                    other.sample_interface.pe
                ][other.sample_interface.interface_idx]
                # TODO: can be relaxed
                if (
                    len(sample_global_tensors.dtensor_groups)
                    == len(other_global_tensor.dtensor_groups)
                    == 1
                ):
                    sample_value = next(
                        iter(sample_global_tensors.dtensor_groups.values())
                    )
                    other_value = next(
                        iter(other_global_tensor.dtensor_groups.values())
                    )
                    if len(sample_value) == len(other_value):
                        shape: list[int] = None
                        for sample_tile, other_tile in zip(sample_value, other_value):
                            if (
                                not sample_tile.tile.dtensor_id
                                == other_tile.tile.dtensor_id
                            ):
                                return None
                            dtensor = global_dtensor[sample_tile.tile.dtensor_id]
                            outer_shape = [1, 1, 1, 1]
                            for i in dtensor.shared_dims:
                                outer_shape[i] = dtensor.size[i]
                            if shape is not None and shape != outer_shape:
                                return None
                            else:
                                shape = outer_shape
                            outer_stride = [1] * 4
                            for i in reversed(range(3)):
                                outer_stride[i] = (
                                    outer_stride[i + 1] * outer_shape[i + 1]
                                )
                            sample_offset = dtensor.offset_map[
                                sample_tile.tile.tensor_tile_label
                            ].to_list()
                            sample_flattened_idx = sum(
                                i * s for i, s in zip(sample_offset, outer_stride)
                            )
                            other_offset = dtensor.offset_map[
                                other_tile.tile.tensor_tile_label
                            ].to_list()
                            other_flattened_idx = sum(
                                i * s for i, s in zip(other_offset, outer_stride)
                            )
                            if other_flattened_idx - sample_flattened_idx != 1:
                                return None
                        new_size_list = current_size.to_list()
                        offset_1, offset_2 = sample_offset, other_offset
                        for i in range(4):
                            new_size_list[i] = (
                                new_size_list[i] + offset_2[i] - offset_1[i]
                            )
                        return Size4D.from_list(new_size_list)
                return None

            def get_pes(self) -> list[str]:
                ret: list[str] = []
                for pe_tile in self.interface_list:
                    ret.append(pe_tile.pe)
                return ret

            def __str__(self):
                return (
                    "["
                    + ", ".join(str(interface) for interface in self.interface_list)
                    + "]"
                )

            def __repr__(self):
                return self.__str__()

        class ContiguousInterface:
            """
            ContiguousInterface always acquire adjacent memory block
            """

            def __init__(self, offset: Offset4D, interface: MulticastInterface):
                self.current_offset: Offset4D = offset
                self.total_size: Size4D = Size4D(1, 1, 1, 1)
                self.interface_list: list[MulticastInterface] = [interface]

            def append(self, offset: Offset4D, other: MulticastInterface) -> bool:
                sample = self.interface_list[-1]
                updated_size = sample._contiguous_data_transfer(other, self.total_size)
                if updated_size is None:
                    return False
                self.interface_list.append(other)
                self.current_offset = offset
                self.total_size = updated_size
                return True

            def __str__(self):
                return "; ".join(str(interface) for interface in self.interface_list)

            def __repr__(self):
                return self.__str__()

        @dataclass(frozen=True)
        class GlobalIODMAPort:
            fifo: FIFO
            connect_interface: list[MulticastInterface]
            size: list[int]
            stride: list[int]
            is_input: bool

        def assign_mem_tile(
            dtype: str,
            interface_list: list[MulticastInterface],
            is_input: bool,
            coalesced_size: Size4D,
            tile_size: Size4D,
            tile_shape: list[int],
        ):
            """
            fixme: maybe too aie-specific?
            Assign a memory tile to the given dtensor tiles.
            If no memory tile is available, return None.
            Else, return the assigned memory tile, the port id to shim, and the port ids to compute.
            """
            send_need = len(interface_list) if is_input else 1
            recv_need = (
                1
                if is_input
                else sum(len(group.interface_list) for group in interface_list)
            )
            send_shape: list[int] = tile_shape if is_input else coalesced_size.to_list()
            recv_shape: list[int] = coalesced_size.to_list() if is_input else tile_shape
            tile_total_size = tile_size.get_total_size()
            if os.getenv("VERBOSE") == "1":
                print(f"send_need: {send_need}, recv_need: {recv_need}")
            assigned_mem_tile = None
            # Attempt to use a new memory tile
            if (
                len(self.exp_used_mem_tiles) < MAX_MEM_TILES
                and send_need <= Config.MEM_MAX_SEND
                and recv_need <= Config.MEM_MAX_RECV
            ):
                assigned_mem_tile = SwitchNode(
                    name=f"{len(self.exp_used_mem_tiles)}_mem_tile",
                    send_port_num=Config.MEM_MAX_SEND,
                    recv_port_num=Config.MEM_MAX_RECV,
                )
                self.exp_used_mem_tiles.append(assigned_mem_tile)
            else:
                # Attempt to use an existing memory tile
                for mem_tile in self.exp_used_mem_tiles:
                    if (
                        len(mem_tile.send_ports) + send_need <= Config.MEM_MAX_SEND
                        and len(mem_tile.recv_ports) + recv_need <= Config.MEM_MAX_RECV
                    ):
                        assigned_mem_tile = mem_tile
                        break
            # Use new ports
            if assigned_mem_tile is not None:
                send_ports, recv_ports = [], []
                for i in range(send_need):
                    port = SwitchNode.Port(
                        id=len(assigned_mem_tile.send_ports),
                        data_shape=send_shape,
                        dtype=dtype,
                        connected_nodes=interface_list[i].get_pes() if is_input else [],
                    )
                    assigned_mem_tile.send_ports.append(port)
                    send_ports.append(port.id)
                if is_input:
                    for i in range(recv_need):
                        port = SwitchNode.Port(
                            id=len(assigned_mem_tile.recv_ports),
                            data_shape=recv_shape,
                            dtype=dtype,
                            connected_nodes=[],
                        )
                        assigned_mem_tile.recv_ports.append(port)
                        recv_ports.append(port.id)
                else:
                    for multicast_interface in interface_list:
                        for pe_interface in multicast_interface.interface_list:
                            port = SwitchNode.Port(
                                id=len(assigned_mem_tile.recv_ports),
                                data_shape=recv_shape,
                                dtype=dtype,
                                connected_nodes=[pe_interface.pe],
                            )
                            assigned_mem_tile.recv_ports.append(port)
                            recv_ports.append(port.id)
                assigned_mem_tile.intra_connect.append(
                    SwitchNode.IntraConnect(
                        send_ports,
                        recv_ports,
                        list(
                            range(
                                0,
                                max(send_need, recv_need) * tile_total_size,
                                tile_total_size,
                            )
                        ),
                    )
                )
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_mem_tile: ", end="")
                    assigned_mem_tile.print()
                return (
                    assigned_mem_tile,
                    recv_ports[0] if is_input else send_ports[0],
                    send_ports if is_input else recv_ports,
                )
            # TODO: port reuse
            return None, -1, []

        def assign_shim_tile(
            mem_tile: SwitchNode,
            mem_port: SwitchNode.Port,
            is_input: bool,
        ):
            assigned_shim_tile = None
            # Attempt to use a new shim tile
            if len(self.exp_used_shim_tiles) < MAX_SHIM_TILES:
                assigned_shim_tile = SwitchNode(
                    name=f"{len(self.exp_used_shim_tiles)}_shim_tile",
                    send_port_num=Config.SHIM_MAX_SEND,
                    recv_port_num=Config.SHIM_MAX_RECV,
                )
                self.exp_used_shim_tiles.append(assigned_shim_tile)
            else:
                for shim_tile in self.exp_used_shim_tiles:
                    if (
                        len(shim_tile.send_ports) + 1 <= Config.SHIM_MAX_SEND
                        and len(shim_tile.recv_ports) + 1 <= Config.SHIM_MAX_RECV
                    ):
                        assigned_shim_tile = shim_tile
                        break
            # Use new ports
            if assigned_shim_tile is not None:
                connected_mem = [mem_tile.name]
                send_port = SwitchNode.Port(
                    id=len(assigned_shim_tile.send_ports),
                    data_shape=mem_port.data_shape,
                    dtype=mem_port.dtype,
                    connected_nodes=connected_mem if is_input else [],
                )
                assigned_shim_tile.send_ports.append(send_port)
                recv_port = SwitchNode.Port(
                    id=len(assigned_shim_tile.recv_ports),
                    data_shape=mem_port.data_shape,
                    dtype=mem_port.dtype,
                    connected_nodes=[] if is_input else connected_mem,
                )
                assigned_shim_tile.recv_ports.append(recv_port)
                mem_port.connected_nodes.append(assigned_shim_tile.name)
                assigned_shim_tile.intra_connect.append(
                    SwitchNode.IntraConnect(
                        send_port_ids=[send_port.id],
                        recv_port_ids=[recv_port.id],
                        offsets=[0],
                    )
                )
                if os.getenv("VERBOSE") == "1":
                    print("\nassigned_shim_tile: ", end="")
                    assigned_shim_tile.print()
                return assigned_shim_tile, recv_port.id if is_input else send_port.id
                # TODO: port reuse
            return None, -1

        mapped_interface: dict[str, dict[int, FIFO]] = {
            i: dict() for i in global_tensors.keys()
        }
        global_io_port: list[GlobalIODMAPort] = []
        for idx, dtensor_tile_group in global_tile_to_func.items():
            dtensor = global_dtensor[idx]
            # transfer tile meta data
            is_input = idx in self.global_inputs
            tile_dtype = dtensor.dtype
            tile_param_type = dtensor.type_as_param
            tile_shape = list(dtensor.size)
            for i in dtensor.shared_dims:
                tile_shape[i] = 1
            tile_size = Size4D.from_list(tile_shape)
            # key: offset specific to dtensor
            unresolved_tile: dict[Offset4D, list[MulticastInterface]] = {}
            in_process: set[PEInterface] = set()
            for (
                dtensor_tile,
                interface_list,
            ) in dtensor_tile_group.dtensor_tile_to_pe_interfaces.items():
                unresolved: set[PEInterface] = set()
                for interface in interface_list:
                    if (
                        interface.interface_idx not in mapped_interface[interface.pe]
                        and interface not in in_process
                    ):
                        unresolved.add(interface)
                        in_process.add(interface)
                multicast_list: list[MulticastInterface] = [
                    MulticastInterface(interface) for interface in unresolved
                ]
                changed = True
                while changed:
                    changed = False
                    new_list = []
                    used = [False] * len(multicast_list)
                    for i in range(len(multicast_list)):
                        if used[i]:
                            continue
                        current = multicast_list[i]
                        for j in range(i + 1, len(multicast_list)):
                            if used[j]:
                                continue
                            if current._equal_data_transfer(multicast_list[j]):
                                current.interface_list.update(
                                    multicast_list[j].interface_list
                                )
                                used[j] = True
                                changed = True
                        new_list.append(current)
                    multicast_list = new_list
                if len(multicast_list) > 0:
                    unresolved_tile[
                        dtensor.offset_map[dtensor_tile.tensor_tile_label]
                    ] = multicast_list
            # coalesced access pattern on dtensor will give a hint
            coalesced_access_pattern, coalesce_info, coalesced_multicast_interfaces = (
                coalesce_memory_access(unresolved_tile)
            )
            print("<<<<< coalesced_multicast_interfaces >>>>>")
            print(coalesced_multicast_interfaces)
            print("===== coalesced_multicast_interfaces =====")
            contiguous_interfaces: list[ContiguousInterface] = []
            for start_offset in coalesced_access_pattern.keys():
                coalesced_interfaces: list[list[MulticastInterface]] = (
                    coalesced_multicast_interfaces[start_offset]
                )
                left = 0
                while left < len(coalesced_interfaces):
                    assert len(coalesced_interfaces[left]) == 1, "TO BE IMPLEMETENED"
                    contiguous: ContiguousInterface = ContiguousInterface(
                        coalesce_info[start_offset][left], coalesced_interfaces[left][0]
                    )
                    right = left + 1
                    while right < len(coalesced_interfaces):
                        assert (
                            len(coalesced_interfaces[right]) == 1
                        ), "TO BE IMPLEMETENED"
                        if contiguous.append(
                            coalesce_info[start_offset][right],
                            coalesced_interfaces[right][0],
                        ):
                            right = right + 1
                        else:
                            break
                    contiguous_interfaces.append(contiguous)
                    left = right
            print("\n<<<<< contiguous_interfaces >>>>>")
            for contiguous_interface in contiguous_interfaces:
                print(contiguous_interface)
            print("===== contiguous_interfaces =====\n")
            for contiguous_interface in contiguous_interfaces:
                interface_list: list[MulticastInterface] = (
                    contiguous_interface.interface_list
                )
                size = contiguous_interface.total_size
                while size.get_total_size() != 0:
                    coalesced_size = Size4D.coalesce(size, tile_size)
                    assigned_mem_tile, port_id, ports_to_compute = assign_mem_tile(
                        tile_dtype,
                        interface_list,
                        is_input,
                        coalesced_size,
                        tile_size,
                        tile_param_type,
                    )
                    if assigned_mem_tile is not None:
                        interface_to_fifo: dict[str, FIFO] = {}
                        for port in ports_to_compute:
                            mem_port_to_compute: SwitchNode.Port = (
                                assigned_mem_tile.send_ports[port]
                                if is_input
                                else assigned_mem_tile.recv_ports[port]
                            )
                            if is_input:
                                dma_fifo = self.experimental_fifo_manager.create_fifo(
                                    src=assigned_mem_tile.name,
                                    dst=mem_port_to_compute.connected_nodes,
                                    data_shape=mem_port_to_compute.data_shape,
                                    dtype=mem_port_to_compute.dtype,
                                )
                            else:
                                assert len(mem_port_to_compute.connected_nodes) == 1
                                dma_fifo = self.experimental_fifo_manager.create_fifo(
                                    src=mem_port_to_compute.connected_nodes[0],
                                    dst=[assigned_mem_tile.name],
                                    data_shape=mem_port_to_compute.data_shape,
                                    dtype=mem_port_to_compute.dtype,
                                )
                            for node in mem_port_to_compute.connected_nodes:
                                assert node not in interface_to_fifo
                                interface_to_fifo[node] = dma_fifo
                            mem_port_to_compute.bind_to_fifo(dma_fifo)
                        mem_port_to_shim = (
                            assigned_mem_tile.recv_ports[port_id]
                            if is_input
                            else assigned_mem_tile.send_ports[port_id]
                        )
                        assigned_shim_tile, shim_port_id = assign_shim_tile(
                            assigned_mem_tile,
                            mem_port_to_shim,
                            is_input,
                        )
                        if assigned_shim_tile is None:
                            print("============")
                            print(interface_list)
                            print("====++++====")
                            for shim_tile in self.exp_used_shim_tiles:
                                shim_tile.print()
                            raise ValueError("Fail to assign shim tile")
                        shim_port_to_mem = (
                            assigned_shim_tile.send_ports[shim_port_id]
                            if is_input
                            else assigned_shim_tile.recv_ports[shim_port_id]
                        )
                        if is_input:
                            dma_fifo = self.experimental_fifo_manager.create_fifo(
                                src=assigned_shim_tile.name,
                                dst=shim_port_to_mem.connected_nodes,
                                data_shape=mem_port_to_compute.data_shape,
                                dtype=mem_port_to_compute.dtype,
                            )
                        else:
                            assert (
                                len(shim_port_to_mem.connected_nodes) == 1
                                and shim_port_to_mem.connected_nodes[0]
                                == assigned_mem_tile.name
                            )
                            dma_fifo = self.experimental_fifo_manager.create_fifo(
                                src=assigned_mem_tile.name,
                                dst=[assigned_shim_tile.name],
                                data_shape=mem_port_to_compute.data_shape,
                                dtype=mem_port_to_compute.dtype,
                            )
                        shim_port_to_mem.bind_to_fifo(dma_fifo)
                        mem_port_to_shim.bind_to_fifo(dma_fifo)
                        global_io_port.append(
                            GlobalIODMAPort(
                                fifo=dma_fifo,
                                connect_interface=interface_list,
                                size=coalesced_size,
                                stride=dtensor.stride,
                                is_input=is_input,
                            )
                        )
                        for interface in interface_list:
                            for pe_interface in interface.interface_list:
                                mapped_interface[pe_interface.pe][
                                    pe_interface.interface_idx
                                ] = interface_to_fifo[pe_interface.pe]
                        break
                    size_cp = size.copy()
                    # keep partitioning until success
                    while True:
                        partitioned_size = partition(size_cp)
                        coalesced_size = Size4D.coalesce(partitioned_size, tile_size)
                        partitioned_interface_list = interface_list[
                            : partitioned_size.get_total_size()
                        ]
                        assigned_mem_tile, port_id, ports_to_compute = assign_mem_tile(
                            tile_dtype,
                            partitioned_interface_list,
                            is_input,
                            coalesced_size,
                            tile_size,
                            tile_param_type,
                        )
                        if assigned_mem_tile is not None:
                            interface_to_fifo: dict[str, FIFO] = {}
                            for port in ports_to_compute:
                                mem_port_to_compute: SwitchNode.Port = (
                                    assigned_mem_tile.send_ports[port]
                                    if is_input
                                    else assigned_mem_tile.recv_ports[port]
                                )
                                if is_input:
                                    dma_fifo = (
                                        self.experimental_fifo_manager.create_fifo(
                                            src=assigned_mem_tile.name,
                                            dst=mem_port_to_compute.connected_nodes,
                                            data_shape=mem_port_to_compute.data_shape,
                                            dtype=mem_port_to_compute.dtype,
                                        )
                                    )
                                else:
                                    assert len(mem_port_to_compute.connected_nodes) == 1
                                    dma_fifo = (
                                        self.experimental_fifo_manager.create_fifo(
                                            src=mem_port_to_compute.connected_nodes[0],
                                            dst=[assigned_mem_tile.name],
                                            data_shape=mem_port_to_compute.data_shape,
                                            dtype=mem_port_to_compute.dtype,
                                        )
                                    )
                                for node in mem_port_to_compute.connected_nodes:
                                    assert node not in interface_to_fifo
                                    interface_to_fifo[node] = dma_fifo
                                mem_port_to_compute.bind_to_fifo(dma_fifo)
                            mem_port_to_shim = (
                                assigned_mem_tile.recv_ports[port_id]
                                if is_input
                                else assigned_mem_tile.send_ports[port_id]
                            )
                            assigned_shim_tile, shim_port_id = assign_shim_tile(
                                assigned_mem_tile,
                                mem_port_to_shim,
                                is_input,
                            )
                            if assigned_shim_tile is None:
                                print("============")
                                print(partitioned_interface_list)
                                print("====++++====")
                                for shim_tile in self.exp_used_shim_tiles:
                                    shim_tile.print()
                                raise ValueError("Fail to assign shim tile")
                            shim_port_to_mem = (
                                assigned_shim_tile.send_ports[shim_port_id]
                                if is_input
                                else assigned_shim_tile.recv_ports[shim_port_id]
                            )
                            if is_input:
                                dma_fifo = self.experimental_fifo_manager.create_fifo(
                                    src=assigned_shim_tile.name,
                                    dst=shim_port_to_mem.connected_nodes,
                                    data_shape=mem_port_to_compute.data_shape,
                                    dtype=mem_port_to_compute.dtype,
                                )
                            else:
                                assert (
                                    len(shim_port_to_mem.connected_nodes) == 1
                                    and shim_port_to_mem.connected_nodes[0]
                                    == assigned_mem_tile.name
                                )
                                dma_fifo = self.experimental_fifo_manager.create_fifo(
                                    src=assigned_mem_tile.name,
                                    dst=[assigned_shim_tile.name],
                                    data_shape=mem_port_to_compute.data_shape,
                                    dtype=mem_port_to_compute.dtype,
                                )
                            shim_port_to_mem.bind_to_fifo(dma_fifo)
                            mem_port_to_shim.bind_to_fifo(dma_fifo)
                            global_io_port.append(
                                GlobalIODMAPort(
                                    fifo=dma_fifo,
                                    connect_interface=partitioned_interface_list,
                                    size=coalesced_size,
                                    stride=dtensor.stride,
                                    is_input=is_input,
                                )
                            )
                            for interface in partitioned_interface_list:
                                for pe_interface in interface.interface_list:
                                    mapped_interface[pe_interface.pe][
                                        pe_interface.interface_idx
                                    ] = interface_to_fifo[pe_interface.pe]
                            break
                        size_cp = partitioned_size
                    size = Size4D.subtract(size, partitioned_size)
                    inc = partitioned_size.get_total_size()
                    interface_list = interface_list[inc:]
            # TODO: some data structure in replace of `self.global_io_dma`
            for io_port in global_io_port:
                interfaces: list[MulticastInterface] = io_port.connect_interface
                # TODO: only support limited cases
                if len(interfaces) == 1:
                    tensor_tile_group = global_tensors[
                        interfaces[0].sample_interface.pe
                    ][interfaces[0].sample_interface.interface_idx]
                    time_offset = 0
                    for live_tensor_tiles in tensor_tile_group.dtensor_groups.values():
                        live_tensor_tiles.sort(key=lambda t: t.first_use)
                        end_time = live_tensor_tiles[-1].last_use
                        tile_idx = 0
                        while tile_idx < len(live_tensor_tiles):
                            dtensor_idx = live_tensor_tiles[tile_idx].tile.dtensor_id
                            dtensor = global_dtensor[dtensor_idx]
                            outer_shape = [1, 1, 1, 1]
                            for i in dtensor.shared_dims:
                                outer_shape[i] = dtensor.size[i]
                            outer_stride = [1] * 4
                            for i in reversed(range(3)):
                                outer_stride[i] = (
                                    outer_stride[i + 1] * outer_shape[i + 1]
                                )
                            size_list = [1,1,1,1]
                            offset = dtensor.offset_map[
                                live_tensor_tiles[tile_idx].tile.tensor_tile_label
                            ].to_list()
                            flattened_idx = sum(
                                i * s for i, s in zip(offset, outer_stride)
                            )
                            while tile_idx + 1 < len(live_tensor_tiles):
                                if (
                                    live_tensor_tiles[tile_idx + 1].tile.dtensor_id
                                    == dtensor_idx
                                ):
                                    next_offset = dtensor.offset_map[
                                        live_tensor_tiles[
                                            tile_idx + 1
                                        ].tile.tensor_tile_label
                                    ].to_list()
                                    flattened_next_idx = sum(
                                        i * s for i, s in zip(next_offset, outer_stride)
                                    )
                                    if flattened_next_idx - flattened_idx == 1:
                                        for i in range(4):
                                            size_list[i] = (
                                                size_list[i] + next_offset[i] - offset[i]
                                            )
                                        tile_idx = tile_idx + 1
                                        flattened_idx = flattened_next_idx
                                        offset = next_offset
                                        break

                            # dma

                else:
                    io_tensors: list[LiveDTensorTileGroup] = []
                    for interface in interfaces:
                        tensor_tile_group = global_tensors[
                            interface.sample_interface.pe
                        ][interface.sample_interface.interface_idx]
                        assert (
                            len(tensor_tile_group.dtensor_groups) == 1
                        ), "TO BE IMPLEMENTED"
                        io_tensors.append(tensor_tile_group)
                    # size: io_port.fifo.data_shape

        return mapped_interface

    def bind_port_to_fifo(self):
        for dma_nodes in zip(self.used_shim_tiles, self.used_mem_tiles):
            for dma_node in dma_nodes:
                for send_port in dma_node.send_ports:
                    dma_fifo = self.fifo_manager.get_or_create_fifo(
                        src=dma_node.name,
                        dst=send_port.connected_nodes,
                        data_shape=send_port.data_shape,
                        dtype=send_port.dtype,
                    )
                    send_port.bind_to_fifo(dma_fifo)
                for recv_port in dma_node.recv_ports:
                    assert (
                        len(recv_port.connected_nodes) <= 1
                    ), "fifo should be single-source"
                    dma_fifo = self.fifo_manager.get_or_create_fifo(
                        src=(
                            recv_port.connected_nodes[0]
                            if len(recv_port.connected_nodes) == 1
                            else None
                        ),
                        dst=[dma_node.name],
                        data_shape=recv_port.data_shape,
                        dtype=recv_port.dtype,
                    )
                    recv_port.bind_to_fifo(dma_fifo)
        if os.getenv("VERBOSE") == "1":
            self.fifo_manager.print()
        # bind function ports to fifos
        for func_name, port_map in self.function_port_map.items():
            self.compute_core_io_experimental[func_name] = {}
            for dtensor, port in port_map.items():
                indices = [i for i, x in enumerate(port.queue) if x == dtensor]
                assert len(indices) == 1
                self.compute_core_io_experimental[func_name][dtensor] = (
                    port.bind_fifo.name,
                    indices[0],
                )

    # ------------------------------------------------------------
    # Compute Tile
    # ------------------------------------------------------------

    def map_core_func_to_physical_tiles(self) -> dict[str, tuple[int, int]]:
        """
        Map the core functions to physical tiles.
        TODO:
            - mapping strategies should be selected by cost
            - careful with nodes with multiple inputs/outputs
              (if ports are exceeded, we should try to assign them to adjacent compute tiles to share local memory)
        """
        core_fucn_mapping: dict[str, tuple[int, int]] = {}
        mesh_shape = self.device_config["mesh"]
        max_row, max_col = mesh_shape
        tile_used = np.zeros(mesh_shape, dtype=bool)
        # connected nodes are grouped into chains when COMPUTE_TILE_WITH_SHARED_MEMORY == 2
        if Config.COMPUTE_TILE_WITH_SHARED_MEMORY == 2:

            class NodeDeque:
                def __init__(self, node_name: str):
                    self.nodes: list[str] = [node_name]

                @staticmethod
                def connect(
                    node_1: str,
                    node_2: str,
                    node_deque1: "NodeDeque",
                    node_deque2: "NodeDeque",
                ):
                    if node_2 == node_deque2.nodes[-1]:
                        node_1, node_2 = node_2, node_1
                        node_deque1, node_deque2 = node_deque2, node_deque1
                    if node_1 == node_deque1.nodes[0]:
                        node_deque1.nodes.reverse()
                    if node_2 == node_deque2.nodes[-1]:
                        node_deque2.nodes.reverse()
                    node_deque1.nodes.extend(node_deque2.nodes)
                    return node_deque1

                def __str__(self):
                    return f"NodeDeque({self.nodes})"

                def __repr__(self):
                    return self.__str__()

            connection_info = self.virtual_computation_graph.get_connections()
            connection_info.sort(key=lambda x: x[0], reverse=True)
            grouped_nodes: dict[str, NodeDeque] = {
                name: NodeDeque(name)
                for name in self.virtual_computation_graph.nodes.keys()
            }
            for connection in connection_info:
                grouped_a, grouped_b = (
                    grouped_nodes[connection[1]],
                    grouped_nodes[connection[2]],
                )
                if grouped_a is None or grouped_b is None:
                    continue
                new_group = NodeDeque.connect(
                    connection[1], connection[2], grouped_a, grouped_b
                )
                grouped_nodes.pop(connection[1])
                grouped_nodes.pop(connection[2])
                grouped_nodes[new_group.nodes[0]] = new_group
                grouped_nodes[new_group.nodes[-1]] = new_group
            # TODO: map nodes according to global io
            # heuristic
            traverse_idx = 0
            sorted_values = [grouped_nodes[key] for key in sorted(grouped_nodes.keys())]
            assigned = set()
            for deque in sorted_values:
                if deque in assigned:
                    continue
                assigned.add(deque)
                head = deque.nodes[0]
                while tile_used[traverse_idx // max_col][traverse_idx % max_col]:
                    traverse_idx += 1
                    if traverse_idx >= max_row * max_col:
                        raise ValueError("Too many nodes")
                col_idx = traverse_idx % max_col
                row_idx = traverse_idx // max_col
                core_fucn_mapping[head] = (row_idx, col_idx)
                tile_used[row_idx][col_idx] = True
                reverse = False
                for node in deque.nodes[1:]:
                    while tile_used[row_idx][col_idx]:
                        if reverse:
                            row_idx -= 1
                            if row_idx < 0:
                                row_idx = 0
                                col_idx += 1
                                reverse = not reverse
                        else:
                            row_idx += 1
                            if row_idx >= max_row:
                                row_idx = max_row - 1
                                col_idx += 1
                                reverse = not reverse
                    core_fucn_mapping[node] = (row_idx, col_idx)
                    tile_used[row_idx][col_idx] = True
            if os.getenv("VERBOSE") == "1":
                print("<<< Mapping >>>")
                for node, (row, col) in core_fucn_mapping.items():
                    print(f"{node}: ({row}, {col})")
                print()
            return core_fucn_mapping
        else:
            raise ValueError("To be implemented")

    # ############################################################
    # AIE Code Generation
    # ############################################################
    def aie_codegen_nightly(
        self,
        core_funcs: list[allo_func_d.FuncOp],
        external_funcs: list[allo_func_d.FuncOp],
        use_external_kernels: dict[str, bool],
    ) -> aie_ir.Module:
        # mapping to physical/logical
        # TODO: co-designed mapping to different types of tiles
        mapped_interface = self.map_data_transfer()
        core_function_mapping = self.map_core_func_to_physical_tiles()
        self.external_functions = ""  # fixme
        for func in external_funcs:
            self.external_functions += format_str(str(func), indent=4)

        wrapper_code = f"""
            module {{
                aie.device({self.device_type}) {{
        """
        wrapper_code += self.external_functions
        wrapper_code += """
                }
            }
        """

        with aie_ir.Context() as ctx, aie_ir.Location.unknown():
            # module wrapper
            self.exp_aie_module = aie_ir.Module.parse(wrapper_code, ctx)
            # find device op: aie.device(device_type)
            device_op = None
            for op in self.exp_aie_module.body.operations:
                if isinstance(op, aie_d.DeviceOp):
                    device_op = op
                    break
            assert device_op is not None, "aie.device not found"
            device_body = device_op.regions[0].blocks[0]
            # insert operations in the device body, before `aie.end``
            end_op = None
            for op in device_body.operations:
                if isinstance(op, aie_d.EndOp):
                    end_op = op
                    break
            assert not end_op is None

            with aie_ir.InsertionPoint(end_op):
                # shim tiles
                for i, shim_tile in enumerate(self.exp_used_shim_tiles):
                    self.exp_tile_map[shim_tile.name] = aie_d.TileOp(col=i, row=0)
                # mem tiles
                for i, mem_tile in enumerate(self.exp_used_mem_tiles):
                    self.exp_tile_map[mem_tile.name] = aie_d.TileOp(col=i, row=1)
                # compute tiles
                for func_name, (row, col) in core_function_mapping.items():
                    self.exp_tile_map[func_name] = aie_d.TileOp(col=col, row=row + 2)
                # define fifos
                # - stream fifos: compute <-> compute
                print(self.streams)
                for stream_name, stream in self.streams.items():
                    self.exp_fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        self.exp_tile_map[stream.src],
                        self.exp_tile_map[stream.dst],
                        depth=stream.type.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.type.shape,
                            get_element_type(str(stream.type.dtype)),
                        ),
                    )
                # - io fifos: shim <-> mem <-> compute
                print(self.experimental_fifo_manager.fifos)
                for dma_fifo in self.experimental_fifo_manager.fifos:
                    self.exp_fifo_map[dma_fifo.name] = aie_d.object_fifo(
                        dma_fifo.name,
                        self.exp_tile_map[dma_fifo.src],
                        [self.exp_tile_map[node] for node in dma_fifo.dst],
                        depth=dma_fifo.depth,
                        datatype=aie_ir.MemRefType.get(
                            dma_fifo.data_shape,
                            get_element_type(str(dma_fifo.dtype)),
                        ),
                    )
                # link fifos: in aie, mem tile serves as the linkages
                for dma_node in self.exp_used_mem_tiles:
                    for connect in dma_node.intra_connect:
                        producer = [
                            self.exp_fifo_map[
                                dma_node.recv_ports[recv_port_id].bind_fifo.name
                            ]
                            for recv_port_id in connect.recv_port_ids
                        ]
                        consumer = [
                            self.exp_fifo_map[
                                dma_node.send_ports[send_port_id].bind_fifo.name
                            ]
                            for send_port_id in connect.send_port_ids
                        ]
                        # fixme: is it possible that both producer and consumer are not single fifo?
                        producer_offset = [] if len(producer) == 1 else connect.offsets
                        consumer_offset = [] if len(consumer) == 1 else connect.offsets
                        aie_d.object_fifo_link(
                            producer, consumer, producer_offset, consumer_offset
                        )
                # compute logic on each compute tile
                for func in core_funcs:
                    func_name = func.attributes["sym_name"].value
                    func_core = aie_d.Core(
                        tile=self.exp_tile_map[func_name],
                        link_with=(
                            "external.o" if use_external_kernels[func_name] else None
                        ),
                    )
                    if self.exp_global_ip is None:
                        self.exp_global_ip = aie_ir.InsertionPoint(func_core)
                    self.exp_build_core_function(
                        func_core,
                        func,
                        self.core_func_args[func_name],
                        mapped_interface[func_name],
                    )

                # runtime sequence
                runtime_seq = aiex_d.RuntimeSequenceOp()
                runtime_args = []
                for i in range(len(self.global_inputs) + len(self.global_outputs)):
                    arg = (
                        self.global_inputs[i]
                        if i in self.global_inputs
                        else self.global_outputs[i]
                    )
                    runtime_args.append(
                        aie_ir.MemRefType.get(
                            arg.shape, get_element_type(str(arg.dtype))
                        )
                    )
                runtime_seq_entry_block = runtime_seq.body.blocks.append(*runtime_args)
                with aie_ir.InsertionPoint(runtime_seq_entry_block):
                    # TODO

                    aie_d.EndOp()

        print(self.exp_aie_module)
        return self.exp_aie_module

    def aie_codegen_experimental(
        self,
        core_funcs: list[allo_func_d.FuncOp],
        external_funcs: list[allo_func_d.FuncOp],
        use_external_kernels: dict[str, bool],
    ) -> aie_ir.Module:
        # mapping to physical/logical
        # TODO: co-designed mapping to different types of tiles
        self.map_global_io_to_physical_tiles()

        if os.getenv("VERBOSE") == "1":
            print("<<< function_port_map >>>")
            for func_name, port_map in self.function_port_map.items():
                print(f"{func_name}:")
                for dtensor, port in port_map.items():
                    print(f"  {dtensor}: {port}")
        self.bind_port_to_fifo()
        core_function_mapping = self.map_core_func_to_physical_tiles()
        for func in external_funcs:
            self.external_functions += format_str(str(func), indent=4)

        wrapper_code = f"""
            module {{
                aie.device({self.device_type}) {{
        """
        wrapper_code += self.external_functions
        wrapper_code += """
                }
            }
        """

        with aie_ir.Context() as ctx, aie_ir.Location.unknown():
            # module wrapper
            self.aie_module = aie_ir.Module.parse(wrapper_code, ctx)
            # find device op: aie.device(device_type)
            device_op = None
            for op in self.aie_module.body.operations:
                if isinstance(op, aie_d.DeviceOp):
                    device_op = op
                    break
            assert device_op is not None, "aie.device not found"
            device_body = device_op.regions[0].blocks[0]
            # insert operations in the device body, before `aie.end``
            end_op = None
            for op in device_body.operations:
                if isinstance(op, aie_d.EndOp):
                    end_op = op
                    break
            assert not end_op is None

            with aie_ir.InsertionPoint(end_op):
                # shim tiles
                for i, shim_tile in enumerate(self.used_shim_tiles):
                    self.tile_map[shim_tile.name] = aie_d.TileOp(col=i, row=0)
                # mem tiles
                for i, mem_tile in enumerate(self.used_mem_tiles):
                    self.tile_map[mem_tile.name] = aie_d.TileOp(col=i, row=1)
                # compute tiles
                for func_name, (row, col) in core_function_mapping.items():
                    self.tile_map[func_name] = aie_d.TileOp(col=col, row=row + 2)
                # define fifos
                # - stream fifos: compute <-> compute
                for stream_name, stream in self.streams.items():
                    self.fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        self.tile_map[stream.src],
                        self.tile_map[stream.dst],
                        depth=stream.type.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.type.shape,
                            get_element_type(str(stream.type.dtype)),
                        ),
                    )
                # - io fifos: shim <-> mem <-> compute
                for dma_fifo in self.fifo_manager.fifo_map.values():
                    if dma_fifo.src is None or len(dma_fifo.dst) == 0:
                        # from/to global
                        continue
                    self.fifo_map[dma_fifo.name] = aie_d.object_fifo(
                        dma_fifo.name,
                        self.tile_map[dma_fifo.src],
                        [self.tile_map[node] for node in dma_fifo.dst],
                        depth=dma_fifo.depth,
                        datatype=aie_ir.MemRefType.get(
                            dma_fifo.data_shape,
                            get_element_type(str(dma_fifo.dtype)),
                        ),
                    )
                # link fifos: in aie, mem tile serves as the linkages
                for dma_node in self.used_mem_tiles:
                    for connect in dma_node.intra_connect:
                        producer = [
                            self.fifo_map[
                                dma_node.recv_ports[recv_port_id].bind_fifo.name
                            ]
                            for recv_port_id in connect.recv_port_ids
                        ]
                        consumer = [
                            self.fifo_map[
                                dma_node.send_ports[send_port_id].bind_fifo.name
                            ]
                            for send_port_id in connect.send_port_ids
                        ]
                        # fixme: is it possible that both producer and consumer are not single fifo?
                        producer_offset = [] if len(producer) == 1 else connect.offsets
                        consumer_offset = [] if len(consumer) == 1 else connect.offsets
                        aie_d.object_fifo_link(
                            producer, consumer, producer_offset, consumer_offset
                        )

                # compute logic on each compute tile
                for func in core_funcs:
                    func_name = func.attributes["sym_name"].value
                    func_core = aie_d.Core(
                        tile=self.tile_map[func_name],
                        link_with=(
                            "external.o" if use_external_kernels[func_name] else None
                        ),
                    )
                    if self.global_ip is None:
                        self.global_ip = aie_ir.InsertionPoint(func_core)
                    self.build_core_function(
                        func_core, func, self.core_func_args[func_name], True
                    )

                # runtime sequence
                runtime_seq = aiex_d.RuntimeSequenceOp()
                runtime_args = []
                for i in range(len(self.global_inputs) + len(self.global_outputs)):
                    arg = (
                        self.global_inputs[i]
                        if i in self.global_inputs
                        else self.global_outputs[i]
                    )
                    runtime_args.append(
                        aie_ir.MemRefType.get(
                            arg.shape, get_element_type(str(arg.dtype))
                        )
                    )

                runtime_seq_entry_block = runtime_seq.body.blocks.append(*runtime_args)
                with aie_ir.InsertionPoint(runtime_seq_entry_block):
                    sorted_tags = sorted(
                        list(self.global_io_dma.keys()), key=CodeGenerator.parse_tag3
                    )
                    launched_dma = []
                    # use sorted tags to determine the data transfer sequence
                    for tag in sorted_tags:
                        for dma in self.global_io_dma[tag]:
                            dma_fifo = self.fifo_map[dma.port.bind_fifo.name]
                            aiex_d.NpuDmaMemcpyNd(
                                metadata=dma_fifo,
                                bd_id=len(launched_dma),
                                mem=runtime_seq_entry_block.arguments[
                                    dma.dtensor.global_id
                                ],
                                offsets=dma.offset,
                                sizes=dma.size,
                                strides=dma.stride,
                                issue_token=True,
                            )
                            launched_dma.append(dma_fifo)
                            if len(launched_dma) == Config.DMA_MAX_BDS:
                                for launched_fifo in launched_dma:
                                    aiex_d.dma_wait(launched_fifo)
                                launched_dma.clear()
                    for launched_fifo in launched_dma:
                        aiex_d.dma_wait(launched_fifo)
                    aie_d.EndOp()

        return self.aie_module

    def aie_codegen(
        self,
        core_func_groups: dict[str, list[allo_func_d.FuncOp]],
        external_funcs: list[allo_func_d.FuncOp],
        inputs,
        outputs,
        use_external_kernels: dict[str, bool],
    ) -> aie_ir.Module:
        """
        Generate an AIE MLIR module.
        """
        io_mapping, mem_tile_num, shim_tile_num = map_global_io(inputs, outputs)
        wrapper_code = f"""
            module {{
                aie.device({self.device_type}) {{
        """

        # fixme: maybe better to resolve this using IR constructor
        for func in external_funcs:
            self.external_functions += format_str(str(func), indent=4)

        wrapper_code += self.external_functions
        wrapper_code += """
                }
            }
        """
        with aie_ir.Context() as ctx, aie_ir.Location.unknown():
            # module wrapper
            self.aie_module = aie_ir.Module.parse(wrapper_code, ctx)
            # find device op: aie.device(device_type)
            device_op = None
            for op in self.aie_module.body.operations:
                if isinstance(op, aie_d.DeviceOp):
                    device_op = op
                    break
            assert device_op is not None, "aie.device not found"
            device_body = device_op.regions[0].blocks[0]
            # insert operations in the device body, before `aie.end``
            end_op = None
            for op in device_body.operations:
                if isinstance(op, aie_d.EndOp):
                    end_op = op
                    break
            assert not end_op is None

            with aie_ir.InsertionPoint(end_op):
                # shim tile
                for shim_id in range(shim_tile_num):
                    self.tile_map[f"shim_{shim_id}"] = aie_d.TileOp(col=shim_id, row=0)
                # mem tiles
                for mem_id in range(mem_tile_num):
                    self.tile_map[f"mem_{mem_id}"] = aie_d.TileOp(col=mem_id, row=1)
                # compute tiles
                # 'logic' mapping for different function groups.
                mappings = {}
                for func_name in core_func_groups:
                    if len(inputs[func_name]["_global"]) > 0:
                        mappings[func_name] = inputs[func_name]["_global"][0].mapping
                    else:
                        mappings[func_name] = outputs[func_name]["_global"][0].mapping
                aie_mesh = self.device_config["mesh"]
                for func_name, tile_ids in map_kernels_to_device_mesh(
                    mappings, aie_mesh
                ).items():
                    for idx, func in zip(tile_ids, core_func_groups[func_name]):
                        func_name = func.attributes["sym_name"].value
                        self.tile_map[f"compute_{func_name}"] = aie_d.TileOp(
                            col=idx[0],
                            row=idx[1] + 2,
                        )

                for io, arg_lst in (("in", inputs), ("out", outputs)):
                    for func_name, sub_func_lst in arg_lst.items():
                        for idx, dtensor in enumerate(sub_func_lst["_global"]):
                            placement = dtensor.global_placement
                            # shim <-> mem (one to one)
                            for dma_tile in io_mapping[dtensor.name]:
                                # define objectfifo
                                name = f"{io}_shim_{dtensor.name}{dma_tile.dtensor_tile_id}"
                                producer = (
                                    self.tile_map[f"shim_{dma_tile.shim_id}"]
                                    if io == "in"
                                    else self.tile_map[f"mem_{dma_tile.mem_id}"]
                                )
                                consumer = (
                                    [self.tile_map[f"mem_{dma_tile.mem_id}"]]
                                    if io == "in"
                                    else [self.tile_map[f"shim_{dma_tile.shim_id}"]]
                                )
                                idx_ = next(
                                    (
                                        i
                                        for i, size in enumerate(dma_tile.size)
                                        if size != 1
                                    ),
                                    None,
                                )
                                if idx_ is None:
                                    shape = [1]
                                else:
                                    shape = dma_tile.size[idx_:]
                                memref_type = aie_ir.MemRefType.get(
                                    shape,
                                    get_element_type(str(dtensor.dtype)),
                                )
                                fifo_shim = self.fifo_map[name] = aie_d.object_fifo(
                                    name,
                                    producer,
                                    consumer,
                                    depth=2,
                                    datatype=memref_type,
                                )

                                # mem <-> compute (one to ?)
                                fifo_mem = []
                                mem_stride = [0]
                                mem_tile = self.tile_map[f"mem_{dma_tile.mem_id}"]
                                local_memref_type = aie_ir.MemRefType.get(
                                    dtensor.type_as_param,
                                    get_element_type(str(dtensor.dtype)),
                                )
                                for tensor_tile in dma_tile.tensor_tile_labels:
                                    # distribute to placement[tensor_tile]
                                    compute_tiles = []
                                    name = f"{io}_mem_{dtensor.name}_{tensor_tile}"
                                    for tile in placement[tensor_tile]:
                                        # some distributed tile do not have global output
                                        if (
                                            dtensor in outputs[func_name][tile]
                                            or dtensor in inputs[func_name][tile]
                                        ):
                                            idx_str = "_".join([str(x) for x in tile])
                                            # seems confusing. "sym_name" is parsed in this way
                                            compute_tiles.append(
                                                self.tile_map[
                                                    f"compute_{func_name}_{idx_str}"
                                                ]
                                            )
                                            self.compute_core_io.setdefault(
                                                f"{func_name}_{idx_str}", {}
                                            )[dtensor] = name
                                    if io == "in":
                                        producer = mem_tile
                                    else:
                                        # fixme: only one valid producer
                                        assert len(compute_tiles) == 1
                                        producer = compute_tiles[0]
                                    consumer = (
                                        compute_tiles if io == "in" else [mem_tile]
                                    )
                                    fifo = self.fifo_map[name] = aie_d.object_fifo(
                                        name,
                                        producer,
                                        consumer,
                                        depth=2,
                                        datatype=local_memref_type,
                                    )
                                    fifo_mem.append(fifo)
                                    mem_stride.append(
                                        mem_stride[-1]
                                        + np.prod(dtensor.get_local_shape())
                                    )
                                aie_d.object_fifo_link(
                                    fifo_shim if io == "in" else fifo_mem,
                                    fifo_mem if io == "in" else fifo_shim,
                                    [] if io == "in" else mem_stride[:-1],
                                    mem_stride[:-1] if io == "in" else [],
                                )
                # compute <-> compute
                for stream_name, stream in self.streams.items():
                    src_tile = self.tile_map[f"compute_{stream.src}"]
                    dst_tile = [self.tile_map[f"compute_{stream.dst}"]]
                    self.fifo_map[stream_name] = aie_d.object_fifo(
                        stream_name,
                        src_tile,
                        dst_tile,
                        depth=stream.type.depth,
                        datatype=aie_ir.MemRefType.get(
                            stream.type.shape,
                            get_element_type(str(stream.type.dtype)),
                        ),
                    )

                for func_name in core_func_groups:
                    for func in core_func_groups[func_name]:
                        func_name_w_id = func.attributes["sym_name"].value
                        func_core = aie_d.Core(
                            tile=self.tile_map[f"compute_{func_name_w_id}"],
                            link_with=(
                                "external.o"
                                if use_external_kernels[func_name_w_id]
                                else None
                            ),
                        )
                        if self.global_ip is None:
                            self.global_ip = aie_ir.InsertionPoint(func_core)
                        self.build_core_function(
                            func_core, func, self.core_func_args[func_name_w_id]
                        )

                # runtime sequence
                runtime_seq = aiex_d.RuntimeSequenceOp()
                runtime_args = []
                for i in range(len(self.global_inputs) + len(self.global_outputs)):
                    arg = (
                        self.global_inputs[i]
                        if i in self.global_inputs
                        else self.global_outputs[i]
                    )
                    runtime_args.append(
                        aie_ir.MemRefType.get(
                            arg.shape, get_element_type(str(arg.dtype))
                        )
                    )

                runtime_seq_entry_block = runtime_seq.body.blocks.append(*runtime_args)
                with aie_ir.InsertionPoint(runtime_seq_entry_block):
                    dma_tiles: list = []
                    bd_cnt = 0
                    for i in range(len(self.global_inputs) + len(self.global_outputs)):
                        io = "in" if i in self.global_inputs else "out"
                        dtensor = (
                            self.global_inputs[i]
                            if i in self.global_inputs
                            else self.global_outputs[i]
                        )

                        for dma_tile in io_mapping[dtensor.name]:
                            dma_fifo = self.fifo_map[
                                f"{io}_shim_{dtensor.name}{dma_tile.dtensor_tile_id}"
                            ]
                            aiex_d.NpuDmaMemcpyNd(
                                metadata=dma_fifo,
                                bd_id=bd_cnt,
                                mem=runtime_seq_entry_block.arguments[i],
                                offsets=dma_tile.offset,
                                sizes=dma_tile.size,
                                strides=dma_tile.stride,
                                issue_token=True,
                            )
                            bd_cnt += 1
                            dma_tiles.append(dma_fifo)
                    # DMA wait
                    for dma_tile in dma_tiles:
                        aiex_d.dma_wait(dma_tile)
                    aie_d.EndOp()

        return self.aie_module
