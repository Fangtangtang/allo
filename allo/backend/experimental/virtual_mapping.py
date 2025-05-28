# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ..._mlir.dialects import func as func_d, allo as allo_d

# ############################################################
# Base Elements
# ############################################################
class VirtualNode:
    def __init__(self, func: func_d.FuncOp):
        self.func_name: str = func.attributes["sym_name"].value
        self.func: func_d.FuncOp = func
        self.inputs: list[str] = []
        self.outputs: list[str] = []

    def add_input(self, name: str):
        self.inputs.append(name)

    def add_output(self, name: str):
        self.outputs.append(name)

    @staticmethod
    def is_isomorphic(node1: "VirtualNode", node2: "VirtualNode"):
        # TODO
        return node1.func_name == node2.func_name


class VirtualEdge:
    def __init__(self, name: str):
        self.name: str = name
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
                    edge = VirtualEdge(stream_name)
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
            edge.src.add_output(edge.name)
            edge.dst.add_input(edge.name)

    def print_graph(self):
        print("\n<<<<< Computation Graph >>>>>")
        for edge in self.edges.values():
            print(f"{edge.src.func_name} -> {edge.dst.func_name}: {edge.name}")
        print("<<<<< Computation Graph >>>>>\n")
