# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from collections import defaultdict, Counter
import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.dialects import func as func_d, allo as allo_d
from .utils import Argument, parse_kernel_name, Stream, StreamType, get_df_kernels
from ...memory import DTensor, Size4D, Offset4D


# ############################################################
# Memory
# ############################################################
@dataclass
class GlobalDTensorTile:
    dtensor_id: int
    tensor_tile_label: str

    def __hash__(self):
        return hash((self.dtensor_id, self.tensor_tile_label))

    def __eq__(self, other):
        return (
            self.dtensor_id == other.dtensor_id
            and self.tensor_tile_label == other.tensor_tile_label
        )

    def __str__(self):
        return f"{self.dtensor_id} ({self.tensor_tile_label})"

    def __repr__(self):
        return f"{self.dtensor_id} ({self.tensor_tile_label})"


class DTensorTileGroup:
    """
    DTensor tiles -> PEs (functions) using the same DTensor tile.
    """

    def __init__(self, order_tag: str):
        self.order_tag = order_tag
        self.dtensor_tile_to_pes: dict[GlobalDTensorTile, list[str]] = {}

    def add_tile(self, tile: GlobalDTensorTile, pe: str):
        if tile not in self.dtensor_tile_to_pes:
            self.dtensor_tile_to_pes[tile] = []
        self.dtensor_tile_to_pes[tile].append(pe)

    def print(self):
        for tile, pes in self.dtensor_tile_to_pes.items():
            print(f"{tile}: {pes}")


class OrderedDTensorTileGroup:
    """
    order_tag -> DTensorTileGroup

    `order_tag` is useful to determine the correct (deadlock-free) order of tile transfer.
    """

    def __init__(self):
        self.dtensor_tile_groups: dict[str, DTensorTileGroup] = {}

    def add_tile(self, tile: GlobalDTensorTile, order_tag: str, pe: str):
        if order_tag not in self.dtensor_tile_groups:
            self.dtensor_tile_groups[order_tag] = DTensorTileGroup(order_tag)
        self.dtensor_tile_groups[order_tag].add_tile(tile, pe)

    def print(self):
        for order_tag, tiles in self.dtensor_tile_groups.items():
            print(f"<<<<< {order_tag} >>>>>")
            tiles.print()


class FIFO:
    def __init__(
        self,
        name: str,
        src: str,
        dst: list[str],
        data_shape: list[int],
        dtype: str,
        depth: int = 2,
    ):
        self.name = name
        self.src = src
        self.dst = sorted(dst)
        self.data_shape = data_shape
        self.dtype = dtype
        self.depth = depth

    def __str__(self):
        return f"FIFO({self.name}, src={self.src}, dst={self.dst}, {self.dtype}{self.data_shape}, depth={self.depth})"

    def __repr__(self):
        return self.__str__()


class FIFOManager:
    def __init__(self):
        self.fifo_map: dict[tuple, FIFO] = {}

    def get_or_create_fifo(
        self, src: str, dst: list[str], data_shape: list[str], dtype: str
    ) -> FIFO:
        dst = sorted(dst)
        key = (src, tuple(dst), tuple(data_shape), dtype)

        if key in self.fifo_map:
            return self.fifo_map[key]
        else:
            fifo = FIFO(
                name=f"fifo_{len(self.fifo_map)}",
                src=src,
                dst=dst,
                data_shape=data_shape,
                dtype=dtype,
            )
            self.fifo_map[key] = fifo
            return fifo

    def print(self):
        print("\n***** FIFOs *****")
        for key, fifo in self.fifo_map.items():
            print(f"{key}: {fifo}")
        print("***** FIFOs *****\n")


class SwitchNode:
    class Port:
        def __init__(
            self, id: int, data_shape: list[int], dtype: str, connected_nodes: list[str]
        ):
            self.id = id
            self.data_shape = data_shape
            self.dtype = dtype
            self.connected_nodes = connected_nodes
            self.bind_fifo = None

        def bind_to_fifo(self, fifo: FIFO):
            assert (
                self.bind_fifo is None
            ), f"Port {self.id} already bound to {self.bind_fifo}"
            self.bind_fifo = fifo

        def __str__(self):
            return f"Port(data_shape={self.data_shape}, dtype={self.dtype}, connected_nodes={self.connected_nodes})"

        def __repr__(self):
            return self.__str__()

    class IntraConnect:
        def __init__(
            self, send_port_ids: list[int], recv_port_ids: list[int], offsets: list[int]
        ):
            self.send_port_ids = send_port_ids  # send_port_id
            self.recv_port_ids = recv_port_ids  # recv_port_id
            self.offsets = offsets

        def __str__(self):
            return f"(send:{self.send_port_ids} <=> recv:{self.recv_port_ids}, offsets={self.offsets})"

        def __repr__(self):
            return self.__str__()

    def __init__(self, name: str, send_port_num: int, recv_port_num: int):
        self.name = name
        self.max_send = send_port_num
        self.max_recv = recv_port_num
        self.send_ports: list[SwitchNode.Port] = []
        self.recv_ports: list[SwitchNode.Port] = []
        # connect send ports to recv ports
        self.intra_connect: list[SwitchNode.IntraConnect] = []

    def print(self):
        print(f"\n<<<<< Switch {self.name} >>>>>")
        print(f"send ports: {self.send_ports}")
        print(f"recv ports: {self.recv_ports}")
        print(f"intra connect: {self.intra_connect}")


# ############################################################
# Computation Mapping Graph
# ############################################################
class OperationTagger:
    def __init__(self) -> None:
        self.tag_map: dict[str, str] = {}
        self.counter = 0

    def get_tag(self, key: str) -> str:
        """Return existing tag or assign a new one if not present."""
        if key not in self.tag_map:
            tag = f"tag_{self.counter}"
            self.tag_map[key] = tag
            self.counter += 1
        return self.tag_map[key]


class Operation:
    def __init__(self, op_tag: str, repeat: int = 1) -> None:
        self.repeat: int = repeat
        self.op_tag: str = op_tag
        self.global_inputs: list[list[GlobalDTensorTile]] = []
        self.global_outputs: list[list[GlobalDTensorTile]] = []
        self.input_streams: list[Stream] = []
        self.output_streams: list[Stream] = []

    def __str__(self) -> str:
        def fmt_nested_list(nested: list[list]) -> str:
            return (
                "["
                + ", ".join(
                    "[" + ", ".join(str(item) for item in sub) + "]" for sub in nested
                )
                + "]"
            )

        return (
            f"Operation(tag='{self.op_tag}', repeat={self.repeat})\n"
            f"\tGlobal Inputs: {fmt_nested_list(self.global_inputs)}\n"
            f"\tGlobal Outputs: {fmt_nested_list(self.global_outputs)}\n"
            f"\tInput Streams: {[str(s) for s in self.input_streams]}\n"
            f"\tOutput Streams: {[str(s) for s in self.output_streams]}"
        )

    def __repr__(self) -> str:
        return self.__str__()


# ------------------------------------------------------------
class NodeBase:
    node_list: list["NodeBase"] = []

    def __init__(self, name: str = None, func: func_d.FuncOp = None):
        self.id = len(NodeBase.node_list)
        NodeBase.node_list.append(self)
        self.name = name if name is not None else f"function_{self.id}"
        self.func: func_d.FuncOp = func
        self.operations: list[Operation] = []

    def is_isomorphic_to(self, other: "NodeBase")->bool:
        # TODO: check in a more robust way
        if self is other:
            return True
        if len(self.operations)!=len(other.operations):
            return False
        for op1, op2 in zip(self.operations, other.operations):
            if op1.op_tag != op2.op_tag:
                return False
            in1 = Counter((s.src, s.type_str) for s in op1.input_streams)
            in2 = Counter((s.src, s.type_str) for s in op2.input_streams)
            if in1 != in2:
                return False

            out1 = Counter((s.src, s.type_str) for s in op1.output_streams)
            out2 = Counter((s.src, s.type_str) for s in op2.output_streams)
            if out1 != out2:
                return False
        return True
    
    def get_global_io(
        self,
    ) -> tuple[list[list[GlobalDTensorTile]], list[list[GlobalDTensorTile]]]:
        global_inputs: list[list[GlobalDTensorTile]] = []
        global_outputs: list[list[GlobalDTensorTile]] = []
        for operation in self.operations:
            op_global_inputs, op_global_outputs = [], []
            for inputs in operation.global_inputs:
                op_global_inputs.extend(inputs)
            for outputs in operation.global_outputs:
                op_global_outputs.extend(outputs)
            global_inputs.append(op_global_inputs)
            global_outputs.append(op_global_outputs)
        return global_inputs, global_outputs

    def get_stream_io(self) -> tuple[list[Stream], list[Stream]]:
        input_streams, output_streams = [], []
        for operation in self.operations:
            input_streams.extend(operation.input_streams)
            output_streams.extend(operation.output_streams)
        return input_streams, output_streams

    def __str__(self) -> str:
        op_strs = "\n    ".join(str(op) for op in self.operations)
        return (
            f"\n<<<<< NodeBase(id={self.id}, name='{self.name}') >>>>>\n"
            f"  Operations:\n    {op_strs if op_strs else '(none)'}"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


class InitialNode(NodeBase):
    def __init__(self, func: func_d.FuncOp, operation: Operation):
        super().__init__(func.attributes["sym_name"].value, func)
        self.operations.append(operation)


class CollocatedNode(NodeBase):
    def __init__(self, name: str = None, func: func_d.FuncOp = None):
        super().__init__(name=name, func=func)


# ------------------------------------------------------------
class ComputationGraph:
    def __init__(
        self,
        allo_module: allo_ir.ir.Module,
        stream_map: dict[str, Stream],
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
    ):
        self.allo_module = allo_module
        self.nodes: dict[str, NodeBase] = {}
        self.edges: dict[str, Stream] = stream_map
        self.func_args = core_func_args
        self.tagger = OperationTagger()

        df_kernels = get_df_kernels(allo_module)
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            tag_key = re.sub(
                r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(func.operation)
            )
            operation = Operation(self.tagger.get_tag(tag_key))
            _, indexes = parse_kernel_name(func_name)
            params = core_func_args[func_name]
            global_inputs: list[GlobalDTensorTile] = []
            global_outputs: list[GlobalDTensorTile] = []
            for argument, is_input in params.values():
                if argument.stream is not None:
                    if is_input:
                        operation.input_streams.append(argument.stream)
                    else:
                        operation.output_streams.append(argument.stream)
                if argument.dtensor is not None:
                    if is_input:
                        global_inputs.append(
                            GlobalDTensorTile(
                                argument.dtensor.global_id,
                                argument.dtensor.PE_tile_id_to_tensor_tile_id(indexes),
                            )
                        )
                    else:
                        global_outputs.append(
                            GlobalDTensorTile(
                                argument.dtensor.global_id,
                                argument.dtensor.PE_tile_id_to_tensor_tile_id(indexes),
                            )
                        )
            operation.global_inputs.append(global_inputs)
            operation.global_outputs.append(global_outputs)
            node = InitialNode(func, operation)
            self.nodes[func_name] = node

    # ------------------------------------------------------------
    # Transformation Primitives
    # ------------------------------------------------------------

    def bundle(self, node_name_list:list[str]):
        assert len(node_name_list)>=2, "bundle at least two nodes"
        node_list:list[NodeBase] = []
        for name in node_name_list:
            assert name in self.nodes, f"Node({name}) not found"
            node_list.append(self.nodes[name])
        sample_node: NodeBase = node_list[0]
        for node in node_list:
            if not sample_node.is_isomorphic_to(node):
                raise ValueError(f"Expect to bundle isomorphic nodes, Node({node.name}) is not isomorphic to Node({sample_node.name})")
        bundled_node = CollocatedNode(name=sample_node.name, func=sample_node.func)
        for operation in sample_node.operations:
            new_operation = Operation(operation.op_tag, 0)
            new_operation.input_streams = list(operation.input_streams)
            new_operation.output_streams = list(operation.output_streams)
            bundled_node.operations.append(new_operation)
        for node in node_list:
            for idx, operation in enumerate(node.operations):
                bundled_node.operations[idx].repeat+=operation.repeat
                bundled_node.operations[idx].global_inputs.extend(operation.global_inputs)
                bundled_node.operations[idx].global_outputs.extend(operation.global_outputs)
        # update stream 
        for name, stream in self.edges.items():
            if stream.src in node_name_list:
                stream.src = bundled_node.name
            if stream.dst in node_name_list:
                stream.dst = bundled_node.name
        # update nodes and remove bundled function 
        for name in node_name_list:
            removed = self.nodes.pop(name)
            if not name == bundled_node.name:
                removed.func.erase()
                self.func_args.pop(name)
        self.nodes[bundled_node.name] = bundled_node
        print(bundled_node)
        print(self.allo_module)
        # TODO: core_func_args?        


    def chain(self, node_name_a: str, node_name_b: str):
        node_a, node_b = self.nodes(node_name_a), self.nodes(node_name_b)
        assert node_a is not None and node_b is not None, "node not found"
        # TODO: chain
   
    def refactor_code(self):
        # TODO: to be implemented
        pass

    # ------------------------------------------------------------
    # Graph Information
    # ------------------------------------------------------------
    def get_node_global_io(
        self,
    ) -> tuple[
        dict[str, list[list[GlobalDTensorTile]]],
        dict[str, list[list[GlobalDTensorTile]]],
    ]:
        # TODO
        global_in: dict[str, list[list[GlobalDTensorTile]]] = {}
        global_out: dict[str, list[list[GlobalDTensorTile]]] = {}

        for name, node in self.nodes.items():
            global_in[name] = list()
            global_out[name] = list()
            inputs, outputs = node.get_global_io()
            assert len(inputs) == 1 and len(outputs) == 1, "To be implemented"
            global_in[name].append(inputs[0])
            global_out[name].append(outputs[0])

        return global_in, global_out

    def get_node_dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {key: set() for key in self.nodes.keys()}
        for stream in self.edges.values():
            dependencies[stream.dst].add(stream.src)
        return dependencies

    def get_connections(self) -> dict[tuple[str, str], int]:
        connections: dict[tuple[str, str], int] = {}
        for stream in self.edges.values():
            id_1, id_2 = self.nodes[stream.src].id, self.nodes[stream.dst].id
            if id_1 > id_2:
                key = (stream.dst, stream.src)
            else:
                key = (stream.src, stream.dst)
            if key in connections:
                connections[key] += 1
            else:
                connections[key] = 1
        connection_info: list[tuple[int, str, str]] = []
        for (name_1, name_2), count in connections.items():
            connection_info.append((count, name_1, name_2))
        return connection_info

    # ------------------------------------------------------------
    # Print Graph
    # ------------------------------------------------------------
    def print_graph(self):
        print("\n<<<<< Computation Graph >>>>>")
        print("\n<<<<< Nodes >>>>>")
        # TODO
        print("\n<<<<< Edges >>>>>")
        # TODO
        print("<<<<< Computation Graph >>>>>\n")
