import re
import os
from ..ip import parse_cpp_function


class ExternalModule:
    """
    User defined external kernel for aie
    """

    def __init__(
        self, top: str, impl_path: str, input_idx: list[int], output_idx: list[int]
    ):
        self.top = top  # identifier
        self.impl_path = impl_path
        self.filename = os.path.basename(impl_path)
        assert self.filename.endswith(
            ".cc"
        ), f"Expected a .cc file, but got: {self.filename}"

        self.input_idx = input_idx
        self.output_idx = output_idx

        with open(self.impl_path, "r", encoding="utf-8") as f:
            code = f.read()
            pattern = r'extern\s+"C"\s*{.*?}'
            matches = re.findall(pattern, code, re.DOTALL)
            all_functions = []
            for block in matches:
                func_pattern = (
                    rf"\b[\w\s\[\]<>,:*&]+?\b{self.top}\s*\([^)]*\)\s*{{[^{{}}]*}}"
                )
                functions = re.findall(func_pattern, block)
                all_functions.extend(functions)
            assert len(all_functions) == 1, "invalid exteranl function"
            self.func_declare = all_functions[0]
            self.args = parse_cpp_function(all_functions[0], self.top)
        assert (self.args is not None) or len(self.args) != len(self.input_idx) + len(
            self.output_idx
        ), f"Failed to parse {self.impl}"
