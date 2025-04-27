from .utils import format_str, format_code
from typing import Dict, List, Tuple

from .._mlir.dialects import (
    allo as allo_d,
    func as func_d,
)

class ModuleNode:
    def __init__(self, device_type:str):
        self.device_type = device_type
        self.external_kernel:List[func_d.FuncOp] = []
        self.compute_kermel:List[ComputeKernelNode] = []
        self.runtime_sequence: RuntimeSequenceNode = None

    def add_private_external_kernel(self, func:func_d.FuncOp):
        self.external_kernel.append(func)

    def to_string(self) -> str:
        header_code = format_str("module {", indent=0)
        header_code += format_str(f"aie.device({self.device_type}) {{", indent=2)
        for externel_kernel in self.external_kernel:
            header_code += format_str(str(externel_kernel), indent=4)

        code = "\n"

        if self.runtime_sequence != None:
            code += self.runtime_sequence.to_string()

        tail_code += format_str("}", indent=2)
        tail_code += "}"
        return header_code + code + tail_code

class ComputeKernelNode:
    def __init__(self):
        pass

    def to_string(self) -> str:
        pass

class RuntimeSequenceNode:
    def __init__(self):
        pass

    def to_string(self) -> str:
        pass