# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from ..._mlir.dialects import func as func_d, allo as allo_d
from ..._mlir.ir import Type
# ############################################################
# Base Elements
# ############################################################
class VirtualNode:
    def __init__(self, func: func_d.FuncOp):
        self.func_name: str = func.attributes["sym_name"].value
        self.func: func_d.FuncOp = func
        self.inputs: dict[str, tuple[Type, str]] = {}
        self.outputs: dict[str, tuple[Type, str]] = {}
        self.op_tag: str = re.sub(r'func\.func\s+@[\w\d_]+(\s*\()', r'func.func\1', str(self.func.operation))

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
class CollocatedNode:
    def __init__(self):
        self.execution_queue: list[set[VirtualNode]] = []
        self.inputs: set[str] = []
        self.outputs: set[str] = []


# ############################################################
# Vritual Mapping Graph
# ############################################################
class ComputationGraph:
    def __init__(
        self,
        stream_info: dict[
            str, list[tuple[str, str]]
        ],  # function naem -> list((stream_name, direction))
        df_kernels: list[func_d.FuncOp],
        stream_types_dict: dict[str, Type],
    ):
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

    def is_isomorphic(self, node1: VirtualNode, node2: VirtualNode) -> bool:
        if node1.op_tag != node2.op_tag:
            return False
        if len(node1.inputs) != len(node2.inputs) or len(node1.outputs) != len(node2.outputs):
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

    def print_graph(self):
        print("\n<<<<< Computation Graph >>>>>")
        for edge in self.edges.values():
            print(f"{edge.src.func_name} -> {edge.dst.func_name}: {edge.name} ({edge.type})")
        print("<<<<< Computation Graph >>>>>\n")

    def check_isomorphic(self):
        checked_nodes = set()
        for node1 in self.nodes.values():
            checked_nodes.add(node1)
            for node2 in self.nodes.values():
                if node2 in checked_nodes:
                    continue
                if self.is_isomorphic(node1, node2):
                    print(f"{node1.func_name} is isomorphic to {node2.func_name}")
