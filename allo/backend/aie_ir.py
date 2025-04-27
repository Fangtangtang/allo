from .utils import format_str, format_code
from typing import Dict, List, Tuple

from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
)

class BufferNode:
    def __init__(self):
        pass

    def to_string(self) -> str:
        pass

class FIFONode:
    def __init__(self):
        pass

    def to_string(self) -> str:
        pass

class ComputeKernelNode:
    def __init__(self, external_exe:str, func:func_d.FuncOp):
        self.external_exe = external_exe
        self.buffers:Dict[str,BufferNode] = {}
        self.fifos:Dict[str,FIFONode] = {}
        pass

    def to_string(self) -> str:
        header_code = ""
        with format_code(indent=6):
            header_code += format_str("%global_c0 = arith.constant 0 : index")
            header_code += format_str("%global_c1 = arith.constant 1 : index")
            header_code += format_str(
                "%c9223372036854775807 = arith.constant 9223372036854775807 : index"
            )
            header_code += format_str(
                "scf.for %arg0 = %global_c0 to %c9223372036854775807 step %global_c1 {"
            )

        code = ""

        tail_code = ""
        with format_code(indent=6):
            tail_code += format_str("}")
            tail_code += format_str("aie.end")
        tail_code += "    }"
        if self.external_exe != None:
            tail_code += f' {{link_with = "{self.external_exe}"}}\n'
        else:
            tail_code += "\n"

        return header_code + code + tail_code

class RuntimeSequenceNode:
    def __init__(self):
        pass

    def to_string(self) -> str:
        pass


class ModuleNode:
    def __init__(self, device_type:str):
        self.device_type = device_type
        self.external_kernel:List[func_d.FuncOp] = []

        self.compute_kermel:List[ComputeKernelNode] = []
        
        self.runtime_sequence: RuntimeSequenceNode = None
        self.body_code = ""

    def add_private_external_kernel(self, func:func_d.FuncOp):
        self.external_kernel.append(func)
    
    def add_compute_core(self, compute_core:ComputeKernelNode):
        self.compute_kermel.append(compute_core)

    def to_string(self) -> str:
        header_code = format_str("module {", indent=0)
        header_code += format_str(f"aie.device({self.device_type}) {{", indent=2)
        for externel_kernel in self.external_kernel:
            header_code += format_str(str(externel_kernel), indent=4)

        code = "\n"
        # TODO: physical mapping

        if self.runtime_sequence != None:
            code += self.runtime_sequence.to_string()

        tail_code = format_str("}", indent=2)
        tail_code += "}"
        return header_code + code + tail_code

