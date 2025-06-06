# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from collections import defaultdict
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
# Collocated Node
# ------------------------------------------------------------
# A collocated node is a set of virtual nodes that is mapped to the same physical PE.
# Can be seen as a logical compute node.
# ############################################################
class CollocatedBaseNode:
    node_list: list["CollocatedBaseNode"] = []

    def __init__(self, name: str):
        self.id = len(CollocatedBaseNode.node_list)
        CollocatedBaseNode.node_list.append(self)
        self.name = name
        self.execution_queue: list[set["CollocatedBaseNode"]] = []
        # TODO: global io for collocated nodes
        # global <-> PE: a list corresponding to the execution queue,
        #               each element is a dict mapping CollocatedBaseNode id to a list of GlobalDTensorTile
        self.global_inputs: list[dict[int, list[GlobalDTensorTile]]] = []
        self.global_outputs: list[dict[int, list[GlobalDTensorTile]]] = []
        # input/output node id -> edge stream type
        self.input_streams: dict[int, list[tuple[StreamType, str]]] = defaultdict(list)
        self.output_streams: dict[int, list[tuple[StreamType, str]]] = defaultdict(list)

    def add_input(self, node_id: int, input_type: StreamType, stream_name: str):
        self.input_streams[node_id].append((input_type, stream_name))

    def add_output(self, node_id: int, output_type: StreamType, stream_name: str):
        self.output_streams[node_id].append((output_type, stream_name))

    def __str__(self):
        _str = f"[{self.id}] {self.name}\n"
        _str += f"\tinput streams: {self.input_streams}\n"
        _str += f"\toutput streams: {self.output_streams}\n"
        _str += f"\tglobal inputs: {self.global_inputs}\n"
        _str += f"\tglobal outputs: {self.global_outputs}\n"
        return _str

    def __repr__(self):
        return self.__str__()


class CollocatedInitialNode(CollocatedBaseNode):
    """
    Virtual node wrapper
    """

    def __init__(self, func: func_d.FuncOp, global_inputs, global_outputs):
        super().__init__(func.attributes["sym_name"].value)
        self.execution_queue.append({self})
        self.global_inputs.append({self.id: global_inputs})
        self.global_outputs.append({self.id: global_outputs})


# ############################################################
# Computation Mapping Graph
# ############################################################
class Operation:
    def __init__(self, op_tag: str) -> None:
        self.repeat: int = 1
        self.op_tag: str = op_tag
        self.global_inputs: list[list[GlobalDTensorTile]] = []
        self.global_outputs: list[list[GlobalDTensorTile]] = []
        self.input_streams: list[Stream] = []
        self.output_streams: list[Stream] = []


# ------------------------------------------------------------
class NodeBase:
    node_list: list["NodeBase"] = []

    def __init__(self, name: str = None, func: func_d.FuncOp = None):
        self.id = len(NodeBase.node_list)
        NodeBase.node_list.append(self)
        self.name = name if name is not None else f"function_{self.id}"
        self.func: func_d.FuncOp = func
        self.operations: list[Operation] = []


class InitialNode(NodeBase):
    def __init__(self, func: func_d.FuncOp):
        super().__init__(func.attributes["sym_name"].value, func)


class CollocatedNode(NodeBase):
    def __init__(self):
        super().__init__()


# ------------------------------------------------------------
class ComputationGraph:
    def __init__(
        self,
        allo_module: allo_ir.ir.Module,
        stream_map: dict[str, Stream],
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
    ):
        self.allo_module = allo_module
        self.collocated_nodes: dict[str, CollocatedBaseNode] = {}
        self.edges: dict[str, Stream] = dict(stream_map)
        df_kernels = get_df_kernels(allo_module)
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            _, indexes = parse_kernel_name(func_name)
            params = core_func_args[func_name]
            global_inputs: list[GlobalDTensorTile] = []
            global_outputs: list[GlobalDTensorTile] = []
            for argument, is_input in params.values():
                if argument.dtensor is None:
                    continue
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
            initial_node = CollocatedInitialNode(func, global_inputs, global_outputs)
            self.collocated_nodes[func_name] = initial_node
        # collect nodes' inputs/outputs
        for edge in self.edges.values():
            assert edge.src is not None and edge.dst is not None
            self.collocated_nodes[edge.dst].add_input(
                self.collocated_nodes[edge.src].id, edge.type, edge.name
            )
            self.collocated_nodes[edge.src].add_output(
                self.collocated_nodes[edge.dst].id, edge.type, edge.name
            )

    # ------------------------------------------------------------
    # Check
    # ------------------------------------------------------------
    # class Checker:
    # TODO

    # def virtual_node_isomorphism_check(
    #     self, node1: VirtualNode, node2: VirtualNode
    # ) -> bool:
    #     if node1.op_tag != node2.op_tag:
    #         return False
    #     if len(node1.input_streams) != len(node2.input_streams) or len(
    #         node1.output_streams
    #     ) != len(node2.output_streams):
    #         return False
    #     for src_mame, (input_type, _) in node1.input_streams.items():
    #         if src_mame not in node2.input_streams:
    #             return False
    #         if node2.input_streams[src_mame][0] != input_type:
    #             return False
    #     for dst_name, (output_type, _) in node1.output_streams.items():
    #         if dst_name not in node2.output_streams:
    #             return False
    #         if node2.output_streams[dst_name][0] != output_type:
    #             return False
    #     return True

    # TODO:
    # Implement a stronger isomorphism check by:
    # - Comparing function signatures (argument/result types)
    # - Comparing computation logic on inputs and outputs
    # This would allow structural isomorphism with renamed variables/blocks.

    # def collocated_node_isomorphism_check(
    #     self, node1: CollocatedBaseNode, node2: CollocatedBaseNode
    # ) -> bool:
    #     if len(node1.execution_queue) != len(node2.execution_queue):
    #         return False
    #     for op1, op2 in zip(node1.execution_queue, node2.execution_queue):
    #         op1_sample = next(iter(op1))
    #         op2_sample = next(iter(op2))
    #         if isinstance(op1_sample, CollocatedInitialNode):
    #             op1_sample, op2_sample = op2_sample, op1_sample
    #         # check op1_sample and op2_sample
    #         if isinstance(op1_sample, CollocatedInitialNode):
    #             # both are initial nodes
    #             if not isinstance(op2_sample, CollocatedInitialNode):
    #                 return False
    #         elif isinstance(op2_sample, CollocatedInitialNode):
    #             # op1 is collocated node, op2 is initial node
    #             if len(op1_sample.execution_queue) != 1:
    #                 return False
    #             if not self.collocated_node_isomorphism_check(
    #                 next(iter(op1_sample.execution_queue[0])), op2_sample
    #             ):
    #                 return False
    #         else:
    #             if not self.collocated_node_isomorphism_check(op1_sample, op2_sample):
    #                 return False
    #     return True

    # ------------------------------------------------------------
    # Transformation Primitives
    # ------------------------------------------------------------

    # def bundle(self, nodes: list[CollocatedBaseNode]) -> CollocatedNode:
    #     """
    #     Merges multiple isomorphic nodes—those with the same input/output pattern
    #     and computation logic—into a single node.
    #     The internal computation remains unchanged.
    #     The merged node executes multiple times to receive inputs and send outputs accordingly.
    #     """

    #     # validity check
    #     sample_node: CollocatedBaseNode = nodes[0]
    #     for node in nodes[1:]:
    #         if not self.collocated_node_isomorphism_check(sample_node, node):
    #             raise ValueError("Nodes are not isomorphic")
    #     # bundle to get a new collocated node
    #     bundle_node = CollocatedNode.bundle(nodes)
    #     return bundle_node

    # def chain(
    #     self, node1: CollocatedBaseNode, node2: CollocatedBaseNode
    # ) -> CollocatedNode:
    #     """
    #     Operates on two nodes—typically in a producer-consumer relationship—into a single node.
    #     The resulting node must still satisfy the constraint of having no more than two global inputs
    #     and two global outputs (or use ports that can be shared by compatible data types).
    #     The computations from both nodes are concatenated in a specific order that respects any dependency between them.
    #     """
    #     # fixme: currently, we only support chain nodes with direct dependencies
    #     if node1.id in node2.input_streams:
    #         producer, consumer = node1, node2
    #     elif node2.id in node1.input_streams:
    #         producer, consumer = node2, node1
    #     else:
    #         raise ValueError("Nodes are not directly connected")
    #     chain_node = CollocatedNode.chain(producer, consumer)
    #     return chain_node

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
        for name, node in self.collocated_nodes.items():
            global_in[name] = list()
            global_out[name] = list()
            assert (
                len(node.global_inputs) == 1
                and len(node.global_inputs[0]) == 1
                and len(node.global_outputs) == 1
                and len(node.global_outputs[0]) == 1
            ), "To be implemented"
            global_in[name].append(list(node.global_inputs[0].values())[0])
            global_out[name].append(list(node.global_outputs[0].values())[0])
        return global_in, global_out

    def get_node_dependencies(self) -> dict[str, set[str]]:
        dependencies: dict[str, set[str]] = {}
        for name, node in self.collocated_nodes.items():
            dependencies[name] = set()
            for node_id in node.input_streams.keys():
                dependencies[name].add(CollocatedBaseNode.node_list[node_id].name)
        return dependencies

    def get_connection(self) -> list[tuple[int, str, str]]:
        """
        return a list of tuples (connection number, node_1, node_2)
        do not consider the direction of the connection
        node_1 and node_2 are the names of the nodes, the order is decided by node_id
        """
        connections: list[tuple[int, str, str]] = []
        nodes = list(self.collocated_nodes.values())
        nodes.sort(key=lambda x: x.id)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                connection_num = 0
                if nodes[i].id in nodes[j].input_streams:
                    connection_num += len(nodes[j].input_streams[nodes[i].id])
                if nodes[j].id in nodes[i].input_streams:
                    connection_num += len(nodes[i].input_streams[nodes[j].id])
                if connection_num > 0:
                    connections.append((connection_num, nodes[i].name, nodes[j].name))
        return connections

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

    # def print_isomorphic(self):
    #     checked_nodes = set()
    #     for node1 in self.nodes.values():
    #         checked_nodes.add(node1)
    #         for node2 in self.nodes.values():
    #             if node2 in checked_nodes:
    #                 continue
    #             if self.virtual_node_isomorphism_check(node1, node2):
    #                 print(f"{node1.func_name} is isomorphic to {node2.func_name}")
