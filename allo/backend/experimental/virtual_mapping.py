# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from ..._mlir.dialects import func as func_d, allo as allo_d
from ..._mlir.ir import Type
from .utils import Argument
from ...memory import DTensor


# ############################################################
# Virtual Mapping Base Elements
# ############################################################
class VirtualNode:
    def __init__(self, func: func_d.FuncOp):
        self.func_name: str = func.attributes["sym_name"].value
        self.func: func_d.FuncOp = func
        # input/output node name -> (stream type, stream name)
        self.inputs: dict[str, tuple[Type, str]] = {}
        self.outputs: dict[str, tuple[Type, str]] = {}
        self.op_tag: str = re.sub(
            r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(self.func.operation)
        )

    def add_input(self, name: str, input_type: Type, src: str):
        self.inputs[src] = (input_type, name)

    def add_output(self, name: str, output_type: Type, dst: str):
        self.outputs[dst] = (output_type, name)


class VirtualEdge:
    def __init__(self, name: str, stream_type: Type):
        self.name: str = name
        self.type: Type = stream_type
        self.src: VirtualNode = None
        self.dst: VirtualNode = None


# ############################################################
# Collocated Node
# ------------------------------------------------------------
# A collocated node is a set of virtual nodes that is mapped to the same physical PE
# ############################################################
class CollocatedBaseNode:
    node_list: list["CollocatedBaseNode"] = []

    def __init__(self):
        self.id = len(CollocatedBaseNode.node_list)
        CollocatedBaseNode.node_list.append(self)
        self.execution_queue: list[set["CollocatedBaseNode"]] = []
        # input/output node id -> edge stream type
        # fixme: may conflict when having multiple input/output corresponding to the same node
        self.inputs: dict[int, Type] = {}
        self.outputs: dict[int, Type] = {}

    def add_input(self, node_id: int, input_type: Type):
        self.inputs[node_id] = input_type

    def add_output(self, node_id: int, output_type: Type):
        self.outputs[node_id] = output_type


class CollocatedInitialNode(CollocatedBaseNode):
    """
    Virtual node wrapper
    """

    def __init__(self, v_node: VirtualNode):
        super().__init__()
        self.v_node = v_node
        self.execution_queue.append({self})


class CollocatedNode(CollocatedBaseNode):
    def __init__(self):
        super().__init__()

    @staticmethod
    def bundle(nodes: list[CollocatedBaseNode]) -> "CollocatedNode":
        bundle_node = CollocatedNode()
        node_set = set(nodes)
        bundle_node.execution_queue.append(node_set)
        bundle_node.inputs = nodes[0].inputs
        bundle_node.outputs = nodes[0].outputs
        return bundle_node

    @staticmethod
    def chain(
        producer: CollocatedBaseNode, consumer: CollocatedBaseNode
    ) -> "CollocatedNode":
        chain_node = CollocatedNode()
        chain_node.execution_queue.append({producer, consumer})
        # fixme: may conflict when producer and consumer have the same input/output
        chain_node.inputs = producer.inputs | consumer.inputs
        chain_node.inputs.pop(producer.id)
        chain_node.outputs = producer.outputs | consumer.outputs
        chain_node.outputs.pop(consumer.id)
        return chain_node


# ############################################################
# Vritual Mapping Graph
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
        global_inputs: dict[int, DTensor],
        global_outputs: dict[int, DTensor],
    ):
        # TODO: global io info
        self.nodes: dict[str, VirtualNode] = {}
        self.edges: dict[str, VirtualEdge] = {}
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            self.nodes[func_name] = VirtualNode(func)
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
        if len(node1.inputs) != len(node2.inputs) or len(node1.outputs) != len(
            node2.outputs
        ):
            return False
        for func_name, (input_type, _) in node1.inputs.items():
            if func_name not in node2.inputs:
                return False
            if node2.inputs[func_name][0] != input_type:
                return False
        for func_name, (output_type, _) in node1.outputs.items():
            if func_name not in node2.outputs:
                return False
            if node2.outputs[func_name][0] != output_type:
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
        # build initial nodes
        for node in self.nodes.values():
            initial_node = CollocatedInitialNode(node)
            self.collocated_nodes[node.func_name] = initial_node
        # connect initial nodes
        for node in self.nodes.values():
            for input_node, input_type in node.inputs.items():
                self.collocated_nodes[node.func_name].add_input(
                    self.collocated_nodes[input_node].id, input_type
                )
            for output_node, output_type in node.outputs.items():
                self.collocated_nodes[node.func_name].add_output(
                    self.collocated_nodes[output_node].id, output_type
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
        if node1.id in node2.inputs:
            producer, consumer = node1, node2
        elif node2.id in node1.inputs:
            producer, consumer = node2, node1
        else:
            raise ValueError("Nodes are not directly connected")
        chain_node = CollocatedNode.chain(producer, consumer)
        return chain_node

    # ------------------------------------------------------------
    # Print Graph
    # ------------------------------------------------------------
    def print_graph(self):
        print("\n<<<<< Computation Graph >>>>>")
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
