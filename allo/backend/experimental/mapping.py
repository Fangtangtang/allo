# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from collections import defaultdict
from ..._mlir.dialects import func as func_d, allo as allo_d
from ..._mlir.ir import Type
from .utils import Argument, parse_kernel_name
from ...memory import DTensor, Size4D, Offset4D


# ############################################################
# Memory
# ############################################################
@dataclass
class GlobalDMATile:
    func_param_idx: int  # parameter idx
    dtensor_id: int
    tensor_tile_label: str

    def __hash__(self):
        return hash((self.func_param_idx, self.dtensor_id, self.tensor_tile_label))

    def __eq__(self, other):
        return (
            self.func_param_idx == other.func_param_idx
            and self.dtensor_id == other.dtensor_id
            and self.tensor_tile_label == other.tensor_tile_label
        )

    def __str__(self):
        return f"{self.dtensor_id} ({self.tensor_tile_label})"

    def __repr__(self):
        return f"{self.dtensor_id} ({self.tensor_tile_label})"


class DMATileGroup:
    """
    DMA tiles -> PEs (functions) using the same DMA tile.
    """

    def __init__(self, order_tag: str):
        self.order_tag = order_tag
        self.dma_tile_to_pes: dict[GlobalDMATile, list[str]] = {}

    def add_tile(self, tile: GlobalDMATile, pe: str):
        if tile not in self.dma_tile_to_pes:
            self.dma_tile_to_pes[tile] = []
        self.dma_tile_to_pes[tile].append(pe)

    def print(self):
        for tile, pes in self.dma_tile_to_pes.items():
            print(f"{tile}: {pes}")


class OrderedDMATileGroup:
    """
    order_tag -> DMATileGroup

    `order_tag` is useful to determine the correct (deadlock-free) order of tile transfer.
    """

    def __init__(self):
        self.dma_tile_groups: dict[str, DMATileGroup] = {}

    def add_tile(self, tile: GlobalDMATile, order_tag: str, pe: str):
        if order_tag not in self.dma_tile_groups:
            self.dma_tile_groups[order_tag] = DMATileGroup(order_tag)
        self.dma_tile_groups[order_tag].add_tile(tile, pe)

    def print(self):
        for order_tag, tiles in self.dma_tile_groups.items():
            print(f"<<<<< {order_tag} >>>>>")
            tiles.print()


class GlobalDMANode:
    class Port:
        def __init__(
            self, id: int, data_shape: Size4D, dtype: str, connected_nodes: list[str]
        ):
            self.id = id
            self.data_shape = data_shape
            self.dtype = dtype
            self.connected_nodes = connected_nodes

        def __str__(self):
            return f"Port(data_shape={self.data_shape}, dtype={self.dtype}, connected_nodes={self.connected_nodes})"

        def __repr__(self):
            return self.__str__()

    class IntraConnect:
        def __init__(
            self, send_ports: list[int], recv_ports: list[int], offsets: list[int]
        ):
            self.send_ports = send_ports  # send_port_id
            self.recv_ports = recv_ports  # recv_port_id
            self.offsets = offsets

        def __str__(self):
            return f"(send:{self.send_ports} <=> recv:{self.recv_ports}, offsets={self.offsets})"

        def __repr__(self):
            return self.__str__()

    def __init__(self, tile_name: str, send_port_num: int, recv_port_num: int):
        self.tile_name = tile_name
        self.max_send = send_port_num
        self.max_recv = recv_port_num
        self.send_ports: list[GlobalDMANode.Port] = []
        self.recv_ports: list[GlobalDMANode.Port] = []
        # connect send ports to recv ports
        self.intra_connect: list[GlobalDMANode.IntraConnect] = []

    def print(self):
        print(f"\n<<<<< DMA Tile {self.tile_name} >>>>>")
        print(f"send ports: {self.send_ports}")
        print(f"recv ports: {self.recv_ports}")
        print(f"intra connect: {self.intra_connect}")


# ############################################################
# Virtual Mapping Base Elements
# ############################################################
class VirtualNode:
    def __init__(self, func: func_d.FuncOp):
        self.func_name: str = func.attributes["sym_name"].value
        self.func: func_d.FuncOp = func
        # global <-> PE
        self.global_inputs: list[GlobalDMATile] = []
        self.global_outputs: list[GlobalDMATile] = []
        # inter-PE: input/output node name -> (stream type, stream name)
        self.input_streams: dict[str, tuple[Type, str]] = {}
        self.output_streams: dict[str, tuple[Type, str]] = {}
        self.op_tag: str = re.sub(
            r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(self.func.operation)
        )

    def add_global_input(self, func_param_idx: int, dtensor: DTensor, indexes):
        self.global_inputs.append(
            GlobalDMATile(
                func_param_idx,
                dtensor.global_id,
                dtensor.PE_tile_id_to_tensor_tile_id(indexes),
            )
        )

    def add_global_output(self, func_param_idx: int, dtensor: DTensor, indexes):
        self.global_outputs.append(
            GlobalDMATile(
                func_param_idx,
                dtensor.global_id,
                dtensor.PE_tile_id_to_tensor_tile_id(indexes),
            )
        )

    def add_input(self, name: str, input_type: Type, src: str):
        self.input_streams[src] = (input_type, name)

    def add_output(self, name: str, output_type: Type, dst: str):
        self.output_streams[dst] = (output_type, name)


class VirtualEdge:
    def __init__(self, name: str, stream_type: Type):
        self.name: str = name
        self.type: Type = stream_type
        self.src: VirtualNode = None
        self.dst: VirtualNode = None


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
        #               each element is a dict mapping CollocatedBaseNode id to a list of GlobalDMATile
        self.global_inputs: list[dict[int, list[GlobalDMATile]]] = []
        self.global_outputs: list[dict[int, list[GlobalDMATile]]] = []
        # input/output node id -> edge stream type
        # fixme: may conflict when having multiple input/output corresponding to the same node
        self.input_streams: dict[int, list[tuple[Type, str]]] = defaultdict(list)
        self.output_streams: dict[int, list[tuple[Type, str]]] = defaultdict(list)

    def add_input(self, node_id: int, input_type: Type, stream_name: str):
        self.input_streams[node_id].append((input_type, stream_name))

    def add_output(self, node_id: int, output_type: Type, stream_name: str):
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

    def __init__(self, v_node: VirtualNode):
        super().__init__(v_node.func_name)
        self.v_node = v_node
        self.execution_queue.append({self})
        self.global_inputs.append({self.id: v_node.global_inputs})
        self.global_outputs.append({self.id: v_node.global_outputs})


class CollocatedNode(CollocatedBaseNode):
    # TODO: to be implemented
    def __init__(self):
        raise NotImplementedError("CollocatedNode is not fully implemented")
        super().__init__(f"collocated_{len(CollocatedBaseNode.node_list)}")

    @staticmethod
    def bundle(nodes: list[CollocatedBaseNode]) -> "CollocatedNode":
        bundle_node = CollocatedNode()
        node_set = set(nodes)
        bundle_node.execution_queue.append(node_set)
        bundle_node.input_streams = nodes[0].input_streams
        bundle_node.output_streams = nodes[0].output_streams
        return bundle_node

    @staticmethod
    def chain(
        producer: CollocatedBaseNode, consumer: CollocatedBaseNode
    ) -> "CollocatedNode":
        chain_node = CollocatedNode()
        chain_node.execution_queue.append({producer, consumer})
        # fixme: may conflict when producer and consumer have the same input/output
        chain_node.input_streams = producer.input_streams | consumer.input_streams
        chain_node.input_streams.pop(producer.id)
        chain_node.output_streams = producer.output_streams | consumer.output_streams
        chain_node.output_streams.pop(consumer.id)
        return chain_node


# ############################################################
# Computation Mapping Graph
# ############################################################
class ComputationGraph:
    def __init__(
        self,
        df_kernels: list[func_d.FuncOp],
        stream_info: dict[
            str, list[tuple[str, str]]
        ],  # function naem -> list((stream_name, direction))
        stream_types_dict: dict[str, Type],
        core_func_args: dict[str, dict[int, tuple[Argument, bool]]],
    ):
        self.nodes: dict[str, VirtualNode] = {}
        self.edges: dict[str, VirtualEdge] = {}
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            _, indexes = parse_kernel_name(func_name)
            self.nodes[func_name] = VirtualNode(func)
            params = core_func_args[func_name]
            for idx, (argument, is_input) in params.items():
                if argument.dtensor is None:
                    continue
                if is_input:
                    self.nodes[func_name].add_global_input(
                        idx, argument.dtensor, indexes
                    )
                else:
                    self.nodes[func_name].add_global_output(
                        idx, argument.dtensor, indexes
                    )
        # construct edges
        for func_name, streams in stream_info.items():
            node = self.nodes.get(func_name)
            if node is None:
                raise ValueError(f"Function {func_name} not found in computation graph")
            for stream_name, direction in streams:
                edge = self.edges.get(stream_name)
                if edge is None:
                    edge = VirtualEdge(stream_name, stream_types_dict[stream_name])
                    self.edges[stream_name] = edge
                if direction == "in":
                    assert edge.dst is None
                    edge.dst = node
                elif direction == "out":
                    assert edge.src is None
                    edge.src = node
                else:
                    raise ValueError(
                        f"Invalid stream direction '{direction}' for {func_name}"
                    )
        # collect nodes' inputs/outputs
        for edge in self.edges.values():
            assert edge.src is not None and edge.dst is not None
            edge.src.add_output(edge.name, edge.type, edge.dst.func_name)
            edge.dst.add_input(edge.name, edge.type, edge.src.func_name)

        self.collocated_nodes: dict[str, CollocatedBaseNode] = {}
        self.initialize_collocated_nodes()

    # ------------------------------------------------------------
    # Check
    # ------------------------------------------------------------
    def virtual_node_isomorphism_check(
        self, node1: VirtualNode, node2: VirtualNode
    ) -> bool:
        if node1.op_tag != node2.op_tag:
            return False
        if len(node1.input_streams) != len(node2.input_streams) or len(
            node1.output_streams
        ) != len(node2.output_streams):
            return False
        for src_mame, (input_type, _) in node1.input_streams.items():
            if src_mame not in node2.input_streams:
                return False
            if node2.input_streams[src_mame][0] != input_type:
                return False
        for dst_name, (output_type, _) in node1.output_streams.items():
            if dst_name not in node2.output_streams:
                return False
            if node2.output_streams[dst_name][0] != output_type:
                return False
        return True

        # TODO:
        # Implement a stronger isomorphism check by:
        # - Comparing function signatures (argument/result types)
        # - Comparing computation logic on inputs and outputs
        # This would allow structural isomorphism with renamed variables/blocks.

    def collocated_node_isomorphism_check(
        self, node1: CollocatedBaseNode, node2: CollocatedBaseNode
    ) -> bool:
        if len(node1.execution_queue) != len(node2.execution_queue):
            return False
        for op1, op2 in zip(node1.execution_queue, node2.execution_queue):
            op1_sample = next(iter(op1))
            op2_sample = next(iter(op2))
            if isinstance(op1_sample, CollocatedInitialNode):
                op1_sample, op2_sample = op2_sample, op1_sample
            # check op1_sample and op2_sample
            if isinstance(op1_sample, CollocatedInitialNode):
                # both are initial nodes
                if not isinstance(op2_sample, CollocatedInitialNode):
                    return False
            elif isinstance(op2_sample, CollocatedInitialNode):
                # op1 is collocated node, op2 is initial node
                if len(op1_sample.execution_queue) != 1:
                    return False
                if not self.collocated_node_isomorphism_check(
                    next(iter(op1_sample.execution_queue[0])), op2_sample
                ):
                    return False
            else:
                if not self.collocated_node_isomorphism_check(op1_sample, op2_sample):
                    return False
        return True

    # ------------------------------------------------------------
    # Transformation Primitives
    # ------------------------------------------------------------
    def initialize_collocated_nodes(self):
        # InitialNode share the same name with the virtual node
        # build initial nodes
        for node in self.nodes.values():
            initial_node = CollocatedInitialNode(node)
            self.collocated_nodes[node.func_name] = initial_node
        # connect initial nodes
        for node in self.nodes.values():
            # stream
            for src_name, (input_type, input_stream_name) in node.input_streams.items():
                self.collocated_nodes[node.func_name].add_input(
                    self.collocated_nodes[src_name].id, input_type, input_stream_name
                )
            for dst_name, (
                output_type,
                output_stream_name,
            ) in node.output_streams.items():
                self.collocated_nodes[node.func_name].add_output(
                    self.collocated_nodes[dst_name].id, output_type, output_stream_name
                )

    def bundle(self, nodes: list[CollocatedBaseNode]) -> CollocatedNode:
        """
        Merges multiple isomorphic nodes—those with the same input/output pattern
        and computation logic—into a single node.
        The internal computation remains unchanged.
        The merged node executes multiple times to receive inputs and send outputs accordingly.
        """

        # validity check
        sample_node: CollocatedBaseNode = nodes[0]
        for node in nodes[1:]:
            if not self.collocated_node_isomorphism_check(sample_node, node):
                raise ValueError("Nodes are not isomorphic")
        # bundle to get a new collocated node
        bundle_node = CollocatedNode.bundle(nodes)
        return bundle_node

    def chain(
        self, node1: CollocatedBaseNode, node2: CollocatedBaseNode
    ) -> CollocatedNode:
        """
        Operates on two nodes—typically in a producer-consumer relationship—into a single node.
        The resulting node must still satisfy the constraint of having no more than two global inputs
        and two global outputs (or use ports that can be shared by compatible data types).
        The computations from both nodes are concatenated in a specific order that respects any dependency between them.
        """
        # fixme: currently, we only support chain nodes with direct dependencies
        if node1.id in node2.input_streams:
            producer, consumer = node1, node2
        elif node2.id in node1.input_streams:
            producer, consumer = node2, node1
        else:
            raise ValueError("Nodes are not directly connected")
        chain_node = CollocatedNode.chain(producer, consumer)
        return chain_node

    # ------------------------------------------------------------
    # Graph Information
    # ------------------------------------------------------------
    def get_node_global_io(
        self,
    ) -> tuple[
        dict[str, list[list[GlobalDMATile]]], dict[str, list[list[GlobalDMATile]]]
    ]:
        # TODO
        global_in: dict[str, list[list[GlobalDMATile]]] = {}
        global_out: dict[str, list[list[GlobalDMATile]]] = {}
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
        for node in self.nodes.values():
            print(
                f"{node.func_name}: input streams: {node.input_streams}, output streams: {node.output_streams}, "
                f"\tglobal inputs: {', '.join([str(dma_tile) for dma_tile in node.global_inputs])}, "
                f"\tglobal outputs: {', '.join([str(dma_tile) for dma_tile in node.global_outputs])}"
            )
        print("\n<<<<< Edges >>>>>")
        for edge in self.edges.values():
            print(
                f"{edge.src.func_name} -> {edge.dst.func_name}: {edge.name} ({edge.type})"
            )
        print("<<<<< Computation Graph >>>>>\n")

    def print_isomorphic(self):
        checked_nodes = set()
        for node1 in self.nodes.values():
            checked_nodes.add(node1)
            for node2 in self.nodes.values():
                if node2 in checked_nodes:
                    continue
                if self.virtual_node_isomorphism_check(node1, node2):
                    print(f"{node1.func_name} is isomorphic to {node2.func_name}")
