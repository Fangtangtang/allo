# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from collections import defaultdict, Counter
import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.ir import InsertionPoint, FunctionType, Value, UnitAttr
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
            self.queue: list = []

        def bind_to_fifo(self, fifo: FIFO):
            assert (
                self.bind_fifo is None
            ), f"Port {self.id} already bound to {self.bind_fifo}"
            self.bind_fifo = fifo

        def __str__(self):
            return f"Port(data_shape={self.data_shape}, dtype={self.dtype}, connected_nodes={self.connected_nodes}, queue={self.queue})"

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

    def get_init_tag(self, key: str) -> str:
        """Return existing tag or assign a new one if not present."""
        if key not in self.tag_map:
            tag = f"tag_{self.counter}"
            self.tag_map[key] = tag
            self.counter += 1
        return self.tag_map[key]


# ------------------------------------------------------------
class NodeBase:
    node_list: list["NodeBase"] = []

    def __init__(
        self,
        name: str = None,
        func: func_d.FuncOp = None,
        tag: str = None,
        repeat: int = 0,
    ):
        self.id = len(NodeBase.node_list)
        NodeBase.node_list.append(self)
        self.name = name if name is not None else f"function_{self.id}"
        self.func: func_d.FuncOp = func
        self.repeat: int = repeat
        self.op_tag: str = tag
        # fixme: better solution for global IO, maybe function argument related
        #  argument will be mapped to ports (fifos) and for bundled nodes, the io data should use exactly same ports
        self.global_inputs: list[list[GlobalDTensorTile]] = []
        self.global_outputs: list[list[GlobalDTensorTile]] = []
        self.input_streams: list[Stream] = []
        self.output_streams: list[Stream] = []

    def is_isomorphic_to(self, other: "NodeBase") -> bool:
        # TODO: check in a more robust way
        if self is other:
            return True
        if self.op_tag != other.op_tag:
            return False
        in1 = Counter((s.src, s.type_str) for s in self.input_streams)
        in2 = Counter((s.src, s.type_str) for s in other.input_streams)
        if in1 != in2:
            return False
        out1 = Counter((s.src, s.type_str) for s in self.output_streams)
        out2 = Counter((s.src, s.type_str) for s in other.output_streams)
        if out1 != out2:
            return False
        return True

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
            f"Node({self.id}) {self.name}"
            f"Operation(tag='{self.op_tag}', repeat={self.repeat})\n"
            f"\tGlobal Inputs: {fmt_nested_list(self.global_inputs)}\n"
            f"\tGlobal Outputs: {fmt_nested_list(self.global_outputs)}\n"
            f"\tInput Streams: {[str(s) for s in self.input_streams]}\n"
            f"\tOutput Streams: {[str(s) for s in self.output_streams]}"
        )

    def __repr__(self) -> str:
        return self.__str__()


class InitialNode(NodeBase):
    def __init__(self, func: func_d.FuncOp, tag: str):
        super().__init__(func.attributes["sym_name"].value, func, tag, 1)


class CollocatedNode(NodeBase):
    def __init__(
        self,
        tag: str,
        name: str = None,
        func: func_d.FuncOp = None,
        repeat: int = 0,
    ):
        super().__init__(name=name, func=func, tag=tag, repeat=repeat)


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
        self.dependencies: dict[str, set[str]] = {}

        df_kernels = get_df_kernels(allo_module)
        # construct nodes
        for func in df_kernels:
            func_name = func.attributes["sym_name"].value
            tag_key = re.sub(
                r"func\.func\s+@[\w\d_]+(\s*\()", r"func.func\1", str(func.operation)
            )
            node = InitialNode(func, self.tagger.get_init_tag(tag_key))
            _, indexes = parse_kernel_name(func_name)
            params = core_func_args[func_name]
            global_inputs: list[GlobalDTensorTile] = []
            global_outputs: list[GlobalDTensorTile] = []
            for argument, is_input in params.values():
                if argument.stream is not None:
                    if is_input:
                        node.input_streams.append(argument.stream)
                    else:
                        node.output_streams.append(argument.stream)
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
            node.global_inputs.append(global_inputs)
            node.global_outputs.append(global_outputs)
            self.nodes[func_name] = node
            self.dependencies[func_name] = set()
        # initiate dependencies
        for stream in self.edges.values():
            self.dependencies[stream.dst].add(stream.src)

    # ------------------------------------------------------------
    # Transformation Primitives
    # ------------------------------------------------------------

    def bundle(self, node_name_list: list[str]):
        """
        [A] [B] [C] [D]  => [A] x 4
        
        TODO: bundled nodes can be safely reordered
        """
        assert len(node_name_list) >= 2, "bundle at least two nodes"
        node_list: list[NodeBase] = []
        for name in node_name_list:
            assert name in self.nodes, f"Node({name}) not found"
            node_list.append(self.nodes[name])
        sample_node: NodeBase = node_list[0]
        for node in node_list:
            if not sample_node.is_isomorphic_to(node):
                raise ValueError(
                    f"Expect to bundle isomorphic nodes, Node({node.name}) is not isomorphic to Node({sample_node.name})"
                )
        bundled_node = CollocatedNode(
            tag=sample_node.op_tag, name=sample_node.name, func=sample_node.func
        )
        bundled_node.input_streams = list(sample_node.input_streams)
        bundled_node.output_streams = list(sample_node.output_streams)
        for node in node_list:
            bundled_node.repeat += node.repeat
            bundled_node.global_inputs.extend(node.global_inputs)
            bundled_node.global_outputs.extend(node.global_outputs)
        # update stream
        for name, stream in self.edges.items():
            if stream.src in node_name_list:
                self.dependencies[stream.dst].pop(stream.src)
                stream.src = bundled_node.name
                self.dependencies[stream.dst].add(bundled_node.name)
            if stream.dst in node_name_list:
                stream.dst = bundled_node.name
                self.dependencies[bundled_node.name].add(stream.src)
        # update nodes and remove bundled function
        for name in node_name_list:
            removed = self.nodes.pop(name)
            if not name == bundled_node.name:
                removed.func.erase()
                self.func_args.pop(name)
                self.dependencies.pop(name)
        self.nodes[bundled_node.name] = bundled_node

    def chain(self, node_name_a: str, node_name_b: str):
        """
        [A] [B] => [[A]-[B]]
        """

        @dataclass
        class BufferizedStream:
            arg_idx_a: int
            arg_idx_b: int
            arg_a: Value
            arg_b: Value
            stream_input: list[Value]
            stream_output: list[Value]

        node_a, node_b = self.nodes.pop(node_name_a), self.nodes.pop(node_name_b)
        param_a, param_b = self.func_args[node_name_a], self.func_args[node_name_b]
        assert node_a is not None and node_b is not None, "node not found"
        if node_name_b in self.dependencies[node_name_a]:
            raise ValueError(
                f"Cannot chain Node({node_name_a}) and Node({node_name_b})"
            )
        # TODO: repeat function
        assert (
            node_a.repeat == node_b.repeat == 1
        ), "Cannot chaining nodes with repeats currently"
        bundled_tag = (
            f"({node_a.op_tag})x{node_a.repeat}-({node_b.op_tag})x{node_b.repeat}"
        )
        chained_node = CollocatedNode(bundled_tag, repeat=1)
        chained_node.global_inputs.extend(node_a.global_inputs)
        chained_node.global_inputs.extend(node_b.global_inputs)
        chained_node.global_outputs.extend(node_a.global_outputs)
        chained_node.global_outputs.extend(node_b.global_outputs)
        bufferized_stream: dict[Stream, BufferizedStream] = {}
        node_a.output_streams = [
            stream for stream in node_a.output_streams if stream.dst != node_name_b
        ]
        kept_streams = []
        for stream in node_b.input_streams:
            if stream.src == node_name_a:
                idx_a, idx_b = -1, -1
                for idx, arg_info in param_a.items():
                    if arg_info[0].stream is not None and arg_info[0].stream == stream:
                        idx_a = idx
                        break
                for idx, arg_info in param_b.items():
                    if arg_info[0].stream is not None and arg_info[0].stream == stream:
                        idx_b = idx
                        break
                assert idx_a >= 0 and idx_b >= 0
                bufferized_stream[stream] = BufferizedStream(
                    idx_a, idx_b, None, None, [], []
                )
            else:
                kept_streams.append(stream)
        node_b.input_streams = kept_streams
        chained_node.input_streams.extend(node_a.input_streams)
        chained_node.input_streams.extend(node_b.input_streams)
        chained_node.output_streams.extend(node_a.output_streams)
        chained_node.output_streams.extend(node_b.output_streams)
        # refactor function
        function_a: func_d.FuncOp = node_a.func
        function_b: func_d.FuncOp = node_b.func
        # - function parameters
        in_types_a: list = function_a.attributes["function_type"].value.inputs
        arg_idx_offset = len(in_types_a)
        in_types_b: list = function_b.attributes["function_type"].value.inputs
        out_types_a = function_a.attributes["function_type"].value.results
        out_types_b = function_b.attributes["function_type"].value.results
        with function_a.context, allo_ir.ir.Location.unknown():
            func_type = FunctionType.get(
                in_types_a + in_types_b, out_types_a + out_types_b
            )
            new_function = func_d.FuncOp(
                chained_node.name,
                func_type,
                ip=InsertionPoint(function_a),
            )
            new_function.attributes["df.kernel"] = UnitAttr.get()
            entry_block = new_function.add_entry_block()
            for old, new in zip(
                function_a.arguments + function_b.arguments, new_function.arguments
            ):
                old.replace_all_uses_with(new)
            for bufferized_stream_info in bufferized_stream.values():
                bufferized_stream_info.arg_a = new_function.arguments[
                    bufferized_stream_info.arg_idx_a
                ]
                bufferized_stream_info.arg_b = new_function.arguments[
                    bufferized_stream_info.arg_idx_b + arg_idx_offset
                ]
            with InsertionPoint(entry_block):
                for func_block in function_a.body:
                    for op in func_block.operations:
                        if isinstance(op, func_d.ReturnOp):
                            assert len(op.operands_) == 0
                            continue
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                for func_block in function_b.body:
                    for op in func_block.operations:
                        new_op = op.clone()
                        for old, new in zip(op.results, new_op.results):
                            old.replace_all_uses_with(new)
                function_a.erase()
                function_b.erase()
            # bufferize streams
            for stream, bufferized_stream_info in bufferized_stream.items():
                stream_puts = [
                    use.owner
                    for use in bufferized_stream_info.arg_a.uses
                    if isinstance(use.owner, allo_d.StreamPutOp)
                ]
                stream_gets = [
                    use.owner
                    for use in bufferized_stream_info.arg_b.uses
                    if isinstance(use.owner, allo_d.StreamGetOp)
                ]
                assert len(stream_puts) == len(stream_gets)
                for i in range(len(stream_puts)):
                    stream_put: allo_d.StreamPutOp = stream_puts[i]
                    stream_get: allo_d.StreamGetOp = stream_gets[i]
                    # TODO: support bufferize stream in branches or even loops
                    assert isinstance(
                        stream_put.parent.opview, func_d.FuncOp
                    ) and isinstance(
                        stream_get.parent.opview, func_d.FuncOp
                    ), "Only support bufferize stream in the main body"
                    put_value = stream_put.operands[-1]
                    get_result = stream_get.result
                    get_result.replace_all_uses_with(put_value)
                    stream_put.erase()
                    stream_get.erase()
                # update argument info
                param_a.pop(bufferized_stream_info.arg_idx_a)
                param_b.pop(bufferized_stream_info.arg_idx_b)
                self.edges.pop(stream.name)
        chained_node.func = new_function
        self.func_args.pop(node_name_a)
        self.func_args.pop(node_name_b)
        self.func_args[chained_node.name] = param_a
        for key, value in param_b.items():
            self.func_args[chained_node.name][arg_idx_offset + key] = value
        dep = self.dependencies.pop(node_name_a)
        dep.update(self.dependencies.pop(node_name_b))
        dep.remove(node_name_a)
        self.dependencies[chained_node.name] = dep
        self.nodes[chained_node.name] = chained_node

    # ------------------------------------------------------------
    # Graph Information
    # ------------------------------------------------------------
    def get_node_global_io(
        self,
    ) -> tuple[
        dict[str, list[list[GlobalDTensorTile]]],
        dict[str, list[list[GlobalDTensorTile]]],
    ]:
        global_in: dict[str, list[list[GlobalDTensorTile]]] = {}
        global_out: dict[str, list[list[GlobalDTensorTile]]] = {}

        for name, node in self.nodes.items():
            global_in[name] = node.global_inputs
            global_out[name] = node.global_outputs

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
