# pylint: disable=import-error, no-name-in-module, c-extension-no-member, too-many-nested-blocks
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import shutil

import aie.ir as aie_ir

import allo._mlir._mlir_libs._mlir as allo_ir
from ..._mlir.dialects import func as allo_func_d
from ..._mlir.ir import Type
from ...passes import analyze_read_write_patterns

from ...memory import DTensor

from ..._mlir.passmanager import PassManager as mlir_pass_manager
from .mlir_codegen import CodeGenerator, Argument, Stream
from .utils import (
    Argument,
    Stream,
    inject_external_kernels,
    get_df_kernels,
    classify_aie_functions,
    classify_aie_functions_experimental,
    codegen_external_kernels,
    read_tensor_from_file,
    codegen_host,
)
from .mapping import ComputationGraph, GlobalDMATile, DMATileGroup, OrderedDMATileGroup


class AIE_MLIRModule:
    # ############################################################
    # Construction
    # ############################################################
    def __init__(
        self,
        module: allo_ir.ir.Module,
        top_func_name: str,
        parameter_list: dict[str, int],
        func_args: dict,
        project_dir: str,
        stream_info: dict,
    ):
        """
        Note: the module is data-driven,
            we need to carefully manage data transfer between 'functions' to avoid deadlocks.
            For example, launching the kernels in topological order.
        """
        # module metadata
        self.project_dir: str = project_dir
        self.allo_module: allo_ir.ir.Module = module
        self.top_func_name: str = top_func_name
        self.module_parameter_list = [
            k for k, _ in sorted(parameter_list.items(), key=lambda item: item[1])
        ]

        self.func_args: dict[str, list[Argument]] = {}
        self.streams: dict[str, Stream] = {}
        self.stream_info: dict[str, dict[str, bool]] = {}
        self._init_func_args(func_args)
        self._init_streams(stream_info)

        # index in top fucntion argument list -> DTensor
        self.global_inputs: dict[int, DTensor] = None
        self.global_outputs: dict[int, DTensor] = None
        # function name -> (argument index -> (argument, is_input))
        self.core_func_args: dict[str, dict[int, tuple[Argument, bool]]] = None

        self.aie_module: aie_ir.Module = None

    def _init_func_args(self, func_args: dict):
        tmp_map: dict = {}
        for func_name, args in func_args.items():
            self.func_args[func_name] = []
            for arg in args:
                if arg in tmp_map:
                    self.func_args[func_name].append(tmp_map[arg])
                elif isinstance(arg, DTensor):
                    argument = Argument(arg, None)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                elif isinstance(arg, str):
                    stream = Stream(arg)
                    self.streams[arg] = stream
                    argument = Argument(None, stream)
                    self.func_args[func_name].append(argument)
                    tmp_map[arg] = argument
                else:
                    raise ValueError(f"Unresolved function argument {arg}")

    def _init_streams(self, stream_info: dict):
        """
        Collect allo.stream information for each function.
        """
        for func_name, info_list in stream_info.items():
            self.stream_info[func_name] = {}
            for name, io in info_list:
                if io == "in":
                    self.streams[name].dst = func_name
                    self.stream_info[func_name][name] = True
                else:
                    self.streams[name].src = func_name
                    self.stream_info[func_name][name] = False

    # ############################################################
    # Build
    # ############################################################
    def _init_virtual_graph(
        self, stream_info: dict, stream_types_dict: dict[str, Type]
    ):
        assert (
            self.core_func_args is not None
            and self.global_inputs is not None
            and self.global_outputs is not None
        ), "Analysis of kernel parameters should be done before initializing virtual graph"
        print(self.core_func_args)
        for idx, dtensor in self.global_inputs.items():
            print(idx, dtensor.global_placement)

        df_kernels = get_df_kernels(self.allo_module)
        self.virtual_computation_graph: ComputationGraph = ComputationGraph(
            df_kernels,
            stream_info,
            stream_types_dict,
            self.core_func_args,
        )
        if os.getenv("VERBOSE") == "1":
            self.virtual_computation_graph.print_graph()

    def analyze_global_io(self) -> tuple[
        dict[int, OrderedDMATileGroup],
        dict[int, OrderedDMATileGroup],
    ]:
        # global inputs/outputs
        global_in, global_out = self.virtual_computation_graph.get_node_global_io()
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
            for func_name, global_io in global_in.items():
                print(func_name, ", ".join([str(dma_tile) for dma_tile in global_io]))
            print()
            for func_name, global_io in global_out.items():
                print(func_name, ", ".join([str(dma_tile) for dma_tile in global_io]))
            print(node_order_tag)

        global_in_tile_to_func: dict[int, OrderedDMATileGroup] = {
            i: OrderedDMATileGroup() for i in self.global_inputs.keys()
        }
        global_out_tile_to_func: dict[int, OrderedDMATileGroup] = {
            i: OrderedDMATileGroup() for i in self.global_outputs.keys()
        }
        for func_name, io_info in global_in.items():
            outer_tag = node_order_tag[func_name]
            for i, dma_tile in enumerate(io_info):
                inner_tag = f"{outer_tag}-{i}"
                for tile_ in dma_tile:
                    global_in_tile_to_func[tile_.global_id].add_tile(
                        tile_, inner_tag, func_name
                    )
        for func_name, io_info in global_out.items():
            outer_tag = node_order_tag[func_name]
            for i, dma_tile in enumerate(io_info):
                inner_tag = f"{outer_tag}-{i}"
                for tile_ in dma_tile:
                    global_out_tile_to_func[tile_.global_id].add_tile(
                        tile_, inner_tag, func_name
                    )

        if os.getenv("VERBOSE") == "1":
            print("\n<<<<<<< global_in_tile_to_func >>>>>>>>")
            for i in global_in_tile_to_func.keys():
                global_in_tile_to_func[i].print()
            print("\n<<<<<<< global_out_tile_to_func >>>>>>>>")
            for i in global_out_tile_to_func.keys():
                global_out_tile_to_func[i].print()

        return global_in_tile_to_func, global_out_tile_to_func

    def analyze_kernel_parameters(self):
        """
        Analyze the parameters of each df.kernel.

        Collected information:
            - self.core_func_args: function name -> (argument index -> (argument, is_input))
            - self.global_inputs: global input argument index -> DTensor
            - self.global_outputs: global output argument index -> DTensor
        """
        # init
        self.core_func_args = {}
        self.global_inputs = {}
        self.global_outputs = {}
        # analyze
        df_kernels = get_df_kernels(self.allo_module)
        for kernel in df_kernels:
            kernel_name = kernel.attributes["sym_name"].value
            self.core_func_args[kernel_name] = {}
            # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
            in_idx_list, out_idx_list = analyze_read_write_patterns(kernel)
            for io_idx_list, io_type in (
                (in_idx_list, "in"),
                (out_idx_list, "out"),
            ):
                for io_idx in io_idx_list:
                    argument: Argument = self.func_args[kernel_name][io_idx]
                    self.core_func_args[kernel_name][io_idx] = (
                        argument,
                        io_type == "in",
                    )
                    if not argument.dtensor is None:
                        argument.dtensor.set_access_pattern()
                        argument.dtensor.type_as_param = kernel.arguments[
                            io_idx
                        ].type.shape
                        global_idx = self.func_args[self.top_func_name].index(argument)
                        if io_type == "in":
                            self.global_inputs[global_idx] = argument.dtensor
                        else:
                            self.global_outputs[global_idx] = argument.dtensor
            # streams
            for i, _ in enumerate(kernel.arguments):
                func_arg = self.func_args[kernel_name][i]
                if (
                    i in self.core_func_args[kernel_name]
                    or func_arg.stream is None  # unused
                ):
                    continue
                self.core_func_args[kernel_name][i] = (
                    func_arg,
                    self.stream_info[kernel_name][func_arg.stream.name],
                )
        # validity check
        for i in range(len(self.global_inputs)):
            assert (
                i in self.global_inputs
            ), "inputs should be the starting arguments of the function"
        for i in range(len(self.global_outputs)):
            assert (
                i + len(self.global_inputs) in self.global_outputs
            ), "outputs should be the ending arguments of the function"

    def build_experimental(
        self,
        stream_info: dict,
        stream_types_dict: dict[str, Type],
        device_type="npu1_4col",
        enable_virtual_mapping: bool = False,
    ):
        build_dir = os.path.join(self.project_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)

        self.analyze_kernel_parameters()
        self._init_virtual_graph(stream_info, stream_types_dict)
        if enable_virtual_mapping:
            # TODO: transformation on virtual map. may modify allo_module here
            # TODO: update streams and core_func_args
            pass
        # TODO
        global_in_tile_to_func, global_out_tile_to_func = self.analyze_global_io()
        # inject external kernels
        use_external_kernels, injected_kernels, include_src = inject_external_kernels(
            self.allo_module, self.top_func_name
        )
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-affine",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        # code generation
        top_func, core_funcs, external_funcs = classify_aie_functions_experimental(
            self.allo_module, self.top_func_name
        )
        code_generator = CodeGenerator(
            device_type,
            self.global_inputs,
            self.global_outputs,
            top_func,
            self.core_func_args,
            self.streams,
            self.virtual_computation_graph,
        )
        self.aie_module = code_generator.aie_codegen_experimental(
            core_funcs,
            external_funcs,
            use_external_kernels,
            global_in_tile_to_func,
            global_out_tile_to_func,
        )
        print(self.aie_module)

    def collect_io(
        self,
        func_groups: dict[str, list[allo_func_d.FuncOp]],
    ) -> tuple[dict, dict]:
        """
        Analyze input/output tensors of each function in the groups.
        Returns dictionaries of input/output DTensors for each function group and core.
        """
        # init
        self.core_func_args = {}
        self.global_inputs = {}
        self.global_outputs = {}
        inputs = {}
        outputs = {}
        for func_name, funcs in func_groups.items():
            inputs[func_name] = {}
            outputs[func_name] = {}
            inputs[func_name]["_global"] = []
            outputs[func_name]["_global"] = []
            for func in funcs:
                func_name_w_id = func.attributes["sym_name"].value
                self.core_func_args[func_name_w_id] = {}
                # [NOTE]: function name implies some kind of mapping from io tensor to 'core's
                func_id = tuple(
                    int(x) for x in func_name_w_id.split(func_name + "_")[-1].split("_")
                )
                # fixme: `analyze_read_write_patterns` considers parameters that are both read and written as outputs
                in_idx, out_idx = analyze_read_write_patterns(func)
                for io_lst, io_idx, io in (
                    (inputs, in_idx, "in"),
                    (outputs, out_idx, "out"),
                ):
                    io_lst[func_name][func_id] = []
                    for idx in io_idx:
                        argument: Argument = self.func_args[func_name_w_id][idx]
                        self.core_func_args[func_name_w_id][idx] = (
                            argument,
                            io == "in",
                        )
                        if not argument.dtensor is None:
                            argument.dtensor.set_access_pattern()
                            argument.dtensor.type_as_param = func.arguments[
                                idx
                            ].type.shape
                            if argument.dtensor not in io_lst[func_name]["_global"]:
                                io_lst[func_name]["_global"].append(argument.dtensor)
                                if io == "in":
                                    self.global_inputs[
                                        self.func_args[self.top_func_name].index(
                                            argument
                                        )
                                    ] = argument.dtensor
                                else:
                                    self.global_outputs[
                                        self.func_args[self.top_func_name].index(
                                            argument
                                        )
                                    ] = argument.dtensor
                            io_lst[func_name][func_id].append(argument.dtensor)
                # streams
                for i, _ in enumerate(func.arguments):
                    func_arg = self.func_args[func_name_w_id][i]
                    if (
                        i in self.core_func_args[func_name_w_id]
                        or func_arg.stream is None  # unused
                    ):
                        continue
                    self.core_func_args[func_name_w_id][i] = (
                        func_arg,
                        self.stream_info[func_name_w_id][func_arg.stream.name],
                    )
        # validity check
        for i in range(len(self.global_inputs)):
            assert (
                i in self.global_inputs
            ), "inputs should be the starting arguments of the function"
        for i in range(len(self.global_outputs)):
            assert (
                i + len(self.global_inputs) in self.global_outputs
            ), "outputs should be the ending arguments of the function"

        return inputs, outputs

    def build(self, device_type="npu1_4col"):
        assert device_type in [
            "npu1_4col",
            "npu1",
        ], "This build method requires at least 4 columns."
        build_dir = os.path.join(self.project_dir, "build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        os.makedirs(build_dir)

        # TODO: maybe use other ways to capture the relationship between DTensor, function group
        _, core_func_groups, _ = classify_aie_functions(
            self.allo_module, self.top_func_name
        )
        inputs, outputs = self.collect_io(core_func_groups)

        # - extract external kernels
        use_external_kernels, injected_kernels, include_src = inject_external_kernels(
            self.allo_module, self.top_func_name
        )
        # record original allo mlir
        with open(
            os.path.join(self.project_dir, "original.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.allo_module))
        # - lower tensor to memref with registered pass
        passes = [
            "func.func(convert-linalg-to-affine-loops),lower-affine",
        ]
        pipeline = f'builtin.module({",".join(passes)})'
        with self.allo_module.context:
            mlir_pass_manager.parse(pipeline).run(self.allo_module.operation)
        top_func, core_func_groups, external_funcs = classify_aie_functions(
            self.allo_module, self.top_func_name
        )
        code_generator = CodeGenerator(
            device_type,
            self.global_inputs,
            self.global_outputs,
            top_func,
            self.core_func_args,
            self.streams,
        )
        self.aie_module = code_generator.aie_codegen(
            core_func_groups, external_funcs, inputs, outputs, use_external_kernels
        )
        with open(
            os.path.join(self.project_dir, "top.mlir"), "w", encoding="utf-8"
        ) as f:
            f.write(str(self.aie_module))
        if len(injected_kernels) > 0:
            kernel_code = codegen_external_kernels(injected_kernels, include_src)
            with open(
                os.path.join(self.project_dir, "external.cc"), "w", encoding="utf-8"
            ) as f:
                f.write(kernel_code)
            cmd = f"cd {self.project_dir} && $PEANO_INSTALL_DIR/bin/clang++ -O2 -v -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -DNDEBUG -I $MLIR_AIE_INSTALL_DIR/include -I $MLIR_AIE_EXTERNAL_KERNEL_DIR/aie2 -c external.cc -o external.o"
            with subprocess.Popen(cmd, shell=True) as process:
                process.wait()
            if process.returncode != 0:
                raise RuntimeError("Failed to compile external kernels.")
        # build mlir-aie
        cmd = f"cd {self.project_dir} && aiecc.py --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=build/final.xclbin --no-xchesscc --no-xbridge --peano ${{PEANO_INSTALL_DIR}} --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to compile the MLIR-AIE code")
        # generate host code
        path = os.path.dirname(__file__)
        path = os.path.join(path, "../../harness/aie")
        os.system(f"cp -r {path}/* {self.project_dir}")
        host_code = codegen_host(self.global_inputs, self.global_outputs)
        with open(
            os.path.join(self.project_dir, "test.cpp"), "w", encoding="utf-8"
        ) as f:
            f.write(host_code)
        cmd = f"cd {self.project_dir}/build && cmake .. -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to build AIE project.")
        return self

    def help(self):
        # print the parameter list of the module
        print("Parameter reference:", self.module_parameter_list)

    def __call__(self, *args):
        for i in range(len(self.global_inputs)):
            with open(
                os.path.join(self.project_dir, f"input{i}.data"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join([str(i) for i in args[i].flatten()]))
        cmd = f"cd {self.project_dir} && ./build/top -x build/final.xclbin -i insts.txt -k MLIR_AIE"
        with subprocess.Popen(cmd, shell=True) as process:
            process.wait()
        if process.returncode != 0:
            raise RuntimeError("Failed to execute AIE code.")
        # TODO: need to complete multiple outputs rules
        result = read_tensor_from_file(
            self.global_outputs[len(args) - 1].dtype,
            args[-1].shape,
            f"{self.project_dir}/output.data",
        )
        # suppose the last argument is output
        args[-1][:] = result
