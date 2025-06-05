# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-branches, too-many-nested-blocks, redefined-variable-type
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import os
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

# =======================

import aie.dialects.aie as aie_d
import aie.dialects.aiex as aiex_d
import aie.dialects.arith as aie_arith_d
import aie.dialects.func as aie_func_d
import aie.dialects.scf as aie_scf_d

import aie.ir as aie_ir

# =======================

import allo._mlir._mlir_libs._mlir as allo_ir
import allo._mlir.dialects._memref_ops_gen as allo_memref_d

from ..._mlir.ir import InsertionPoint

from ..utils import format_str
from ..._mlir.dialects import func as allo_func_d
from ...memory import DTensor, Offset4D, Size4D, coalesce_memory_access

from .utils import get_element_type, device_config_map, Argument, Stream, Config
from ..aie import map_kernels_to_device_mesh
from .mapping import (
    GlobalDMANode,
    OrderedDMATileGroup,
    DMATileGroup,
    ComputationGraph,
    DMAFIFOManager,
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
        self.external_functions: str = ""

        # ------------------------------------------------------------
        # Experimental
        # ------------------------------------------------------------
        self.used_mem_tiles: list[GlobalDMANode] = None
        self.used_shim_tiles: list[GlobalDMANode] = None
        self.global_io_dma: dict[str, list[CodeGenerator.GlobalIODMA]] = None
        self.function_port_map: dict[str, dict[DTensor, GlobalDMANode.Port]] = (
            defaultdict(lambda: defaultdict(GlobalDMANode.Port))
        )
        self.fifo_manager: DMAFIFOManager = DMAFIFOManager()
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
            for idx, arg_info in func_args.items():
                if arg_info[0].stream is not None:
                    func_inputs[idx] = arg_info[0].stream.allo_element_type
            func_type = allo_func_d.FunctionType.get(
                func_inputs,
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
                io_map = (
                    self.compute_core_io[parsed_function.name.value]
                    if parsed_function.name.value in self.compute_core_io
                    else {}
                )
                for i, argument in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if arg_info[0].dtensor is not None:
                        acquired = self.fifo_map[io_map[arg_info[0].dtensor]].acquire(
                            1 if arg_info[1] else 0, 1
                        )
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
                for i, _ in enumerate(parsed_function.arguments):
                    if not i in func_args:
                        continue
                    arg_info: tuple[Argument, bool] = func_args[i]
                    if not arg_info[0].dtensor is None:
                        self.fifo_map[io_map[arg_info[0].dtensor]].release(
                            1 if arg_info[1] else 0, 1
                        )

                aie_scf_d.YieldOp([])
            aie_d.EndOp()

    @dataclass(frozen=True)
    class GlobalIODMA:
        dtensor: DTensor
        port: GlobalDMANode.Port
        offset: list[int]
        size: list[int]
        stride: list[int]
        is_input: bool

    @staticmethod
    def parse_tag(tag: str) -> tuple[int, int]:
        outer_tag, inner_tag = tag.split("-")
        return int(outer_tag), int(inner_tag)

    def map_global_io_to_physical_tiles(
        self,
        global_in_tile_to_func: dict[int, OrderedDMATileGroup],
        global_out_tile_to_func: dict[int, OrderedDMATileGroup],
    ) -> tuple[
        list[GlobalDMANode],
        list[GlobalDMANode],
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
        ) -> tuple[GlobalDMANode, int, list[int]]:
            """
            fixme: maybe too aie-specific?
            Assign a memory tile to the given dtensor tiles.
            If no memory tile is available, return None.
            Else, return the assigned memory tile, the port id to shim, and the port ids to compute.
            """
            send_need = len(connected_nodes) if is_input else 1
            recv_need = 1 if is_input else sum(len(group) for group in connected_nodes)
            send_size: list[int] = tile_shape if is_input else coalesced_size.to_list()
            recv_size: list[int] = coalesced_size.to_list() if is_input else tile_shape
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
                assigned_mem_tile = GlobalDMANode(
                    tile_name=f"{len(self.used_mem_tiles)}_mem_tile",
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
                    port = GlobalDMANode.Port(
                        id=len(assigned_mem_tile.send_ports),
                        data_shape=send_size,
                        dtype=dtype,
                        connected_nodes=connected_nodes[i] if is_input else [],
                    )
                    assigned_mem_tile.send_ports.append(port)
                    send_ports.append(port.id)
                if is_input:
                    for i in range(recv_need):
                        port = GlobalDMANode.Port(
                            id=len(assigned_mem_tile.recv_ports),
                            data_shape=recv_size,
                            dtype=dtype,
                            connected_nodes=[],
                        )
                        assigned_mem_tile.recv_ports.append(port)
                        recv_ports.append(port.id)
                else:
                    for group in connected_nodes:
                        for node in group:
                            port = GlobalDMANode.Port(
                                id=len(assigned_mem_tile.recv_ports),
                                data_shape=recv_size,
                                dtype=dtype,
                                connected_nodes=[node],
                            )
                            assigned_mem_tile.recv_ports.append(port)
                            recv_ports.append(port.id)
                assigned_mem_tile.intra_connect.append(
                    GlobalDMANode.IntraConnect(
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
            mem_tile: GlobalDMANode,
            port_id: int,
            is_input: bool,
        ) -> tuple[GlobalDMANode, int]:
            port: GlobalDMANode.Port = (
                mem_tile.recv_ports[port_id]
                if is_input
                else mem_tile.send_ports[port_id]
            )
            assigned_shim_tile = None
            # Attempt to use a new shim tile
            if len(self.used_shim_tiles) < MAX_SHIM_TILES:
                assigned_shim_tile = GlobalDMANode(
                    tile_name=f"{len(self.used_shim_tiles)}_shim_tile",
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
                connected_mem = [mem_tile.tile_name]
                send_port = GlobalDMANode.Port(
                    id=len(assigned_shim_tile.send_ports),
                    data_shape=port.data_shape,
                    dtype=port.dtype,
                    connected_nodes=connected_mem if is_input else [],
                )
                assigned_shim_tile.send_ports.append(send_port)
                recv_port = GlobalDMANode.Port(
                    id=len(assigned_shim_tile.recv_ports),
                    data_shape=port.data_shape,
                    dtype=port.dtype,
                    connected_nodes=[] if is_input else connected_mem,
                )
                assigned_shim_tile.recv_ports.append(recv_port)
                port.connected_nodes.append(assigned_shim_tile.tile_name)
                assigned_shim_tile.intra_connect.append(
                    GlobalDMANode.IntraConnect(
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
            dtensor: DTensor, ordered_tile_group: OrderedDMATileGroup, is_input: bool
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

            def register_function_param_port(port: GlobalDMANode.Port):
                for node in port.connected_nodes:
                    self.function_port_map[node][dtensor] = port

            tile_dtype = dtensor.dtype
            tile_shape = dtensor.size
            for i in dtensor.shared_dims:
                tile_shape[i] = 1
            tile_size = Size4D.from_list(tile_shape)
            # Tags sorted in lexicographic order are used to preserve the data transfer sequence.
            # tiles in DMATileGroup with the same tage can be sent in parallel.
            sorted_tags = sorted(
                list(ordered_tile_group.dma_tile_groups.keys()),
                key=CodeGenerator.parse_tag,
            )
            idx = 0
            while idx < len(sorted_tags):
                update = 0
                tag = sorted_tags[idx]
                offset_map: dict[Offset4D, list[str]] = {}
                # fixme: this is an ugly and problematic hack. We need more elegant and robust way to handle this.
                while len(
                    offset_map
                ) < Config.IO_TILE_LOSE_FACTOR and idx + update < len(sorted_tags):
                    dma_tile_group = ordered_tile_group.dma_tile_groups[
                        sorted_tags[idx + update]
                    ]
                    for dma_tile in dma_tile_group.dma_tile_to_pes.keys():
                        if (
                            dtensor.offset_map[dma_tile.tensor_tile_label]
                            not in offset_map
                        ):
                            offset_map[
                                dtensor.offset_map[dma_tile.tensor_tile_label]
                            ] = []
                        offset_map[
                            dtensor.offset_map[dma_tile.tensor_tile_label]
                        ].extend(dma_tile_group.dma_tile_to_pes[dma_tile])
                    update += 1
                coalesced_access, coalesce_info = coalesce_memory_access(
                    list(offset_map.keys())
                )
                if os.getenv("VERBOSE") == "1":
                    print()
                    print(offset_map)
                    print("access:", coalesced_access)
                    print("=== coalesce_info ===", coalesce_info)
                offset_id = 0
                for offset, size in coalesced_access.items():
                    connected_nodes: list[list[str]] = []
                    for node in coalesce_info[offset]:
                        connected_nodes.append(offset_map[node])
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
                            for port in ports_to_compute:
                                register_function_param_port(
                                    (
                                        assigned_mem_tile.send_ports[port]
                                        if is_input
                                        else assigned_mem_tile.recv_ports[port]
                                    ),
                                )
                            assigned_shim_tile, shim_port_id = assign_shim_tile(
                                assigned_mem_tile,
                                port_id,
                                is_input,
                            )
                            if assigned_shim_tile is None:
                                raise ValueError("Fail to assign shim tile")
                            self.global_io_dma[tag].append(
                                CodeGenerator.GlobalIODMA(
                                    dtensor=dtensor,
                                    port=(
                                        assigned_shim_tile.send_ports[shim_port_id]
                                        if is_input
                                        else assigned_shim_tile.recv_ports[shim_port_id]
                                    ),
                                    offset=coalesce_info[offset][offset_id].to_list(),
                                    size=coalesced_size.to_list(),
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
                                for port in ports_to_compute:
                                    register_function_param_port(
                                        (
                                            assigned_mem_tile.send_ports[port]
                                            if is_input
                                            else assigned_mem_tile.recv_ports[port]
                                        ),
                                    )
                                assigned_shim_tile, shim_port_id = assign_shim_tile(
                                    assigned_mem_tile,
                                    port_id,
                                    is_input,
                                )
                                if assigned_shim_tile is None:
                                    raise ValueError("Fail to assign shim tile")
                                self.global_io_dma[tag].append(
                                    CodeGenerator.GlobalIODMA(
                                        dtensor=dtensor,
                                        port=(
                                            assigned_shim_tile.send_ports[shim_port_id]
                                            if is_input
                                            else assigned_shim_tile.recv_ports[
                                                shim_port_id
                                            ]
                                        ),
                                        offset=coalesce_info[offset][
                                            offset_id
                                        ].to_list(),
                                        size=coalesced_size.to_list(),
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

        for input_idx, ordered_tile_group in global_in_tile_to_func.items():
            map_dtensor_to_physical_tiles(
                self.global_inputs[input_idx],
                ordered_tile_group,
                is_input=True,
            )
        for output_idx, ordered_tile_group in global_out_tile_to_func.items():
            map_dtensor_to_physical_tiles(
                self.global_outputs[output_idx],
                ordered_tile_group,
                is_input=False,
            )
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

    def bind_port_to_fifo(self):
        for dma_nodes in zip(self.used_shim_tiles, self.used_mem_tiles):
            for dma_node in dma_nodes:
                # fixme: fifo with only one src
                for send_port in dma_node.send_ports:
                    dma_fifo = self.fifo_manager.get_or_create_fifo(
                        src=[dma_node.tile_name],
                        dst=send_port.connected_nodes,
                        data_shape=send_port.data_shape,
                        dtype=send_port.dtype,
                    )
                    send_port.bind_to_fifo(dma_fifo)
                for recv_port in dma_node.recv_ports:
                    dma_fifo = self.fifo_manager.get_or_create_fifo(
                        src=recv_port.connected_nodes,
                        dst=[dma_node.tile_name],
                        data_shape=recv_port.data_shape,
                        dtype=recv_port.dtype,
                    )
                    recv_port.bind_to_fifo(dma_fifo)
        if os.getenv("VERBOSE") == "1":
            self.fifo_manager.print()
        # bind function ports to fifos
        for func_name, port_map in self.function_port_map.items():
            self.compute_core_io[func_name] = {}
            for dtensor, port in port_map.items():
                self.compute_core_io[func_name][dtensor] = port.bind_fifo.name

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

            connection_info = self.virtual_computation_graph.get_connection()
            connection_info.sort(key=lambda x: x[0], reverse=True)
            grouped_nodes: dict[str, NodeDeque] = {
                name: NodeDeque(name)
                for name in self.virtual_computation_graph.collocated_nodes.keys()
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
    def aie_codegen_experimental(
        self,
        core_funcs: list[allo_func_d.FuncOp],
        external_funcs: list[allo_func_d.FuncOp],
        use_external_kernels: dict[str, bool],
        global_in_tile_to_func: dict[int, OrderedDMATileGroup],
        global_out_tile_to_func: dict[int, OrderedDMATileGroup],
    ) -> aie_ir.Module:
        # mapping to physical/logical
        # TODO: co-designed mapping to different types of tiles
        self.map_global_io_to_physical_tiles(
            global_in_tile_to_func, global_out_tile_to_func
        )

        if os.getenv("VERBOSE") == "1":
            print("<<< function_port_map >>>")
            for func_name, port_map in self.function_port_map.items():
                print(f"{func_name}:")
                for dtensor, port in port_map.items():
                    print(f"  {dtensor}: {port}")

        self.bind_port_to_fifo()
        core_function_mapping = self.map_core_func_to_physical_tiles()
        # fixme: maybe better to resolve this using IR constructor
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
                    self.tile_map[shim_tile.tile_name] = aie_d.TileOp(col=i, row=0)
                # mem tiles
                for i, mem_tile in enumerate(self.used_mem_tiles):
                    self.tile_map[mem_tile.tile_name] = aie_d.TileOp(col=i, row=1)
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
                    if len(dma_fifo.src) == 0 or len(dma_fifo.dst) == 0:
                        # from/to global
                        continue
                    self.fifo_map[dma_fifo.name] = aie_d.object_fifo(
                        dma_fifo.name,
                        self.tile_map[dma_fifo.src[0]],
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
                        func_core, func, self.core_func_args[func_name]
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
                        list(self.global_io_dma.keys()), key=CodeGenerator.parse_tag
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
