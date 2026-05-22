# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._allo_ops_gen import *
from ._allo_ops_gen import _Dialect
from .._mlir_libs._allo.allo import *
from .func import FuncOp

try:
    from ..ir import (
        ArrayAttr,
        DenseI64ArrayAttr,
        IntegerAttr,
        MemRefType,
        InsertionPoint,
        StringAttr,
        FunctionType,
    )
    from ._ods_common import _cext as _ods_cext
    from ..extras.types import i64
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class GridMapOp(GridMapOp):
    interface_attr = "interface"  # i: input, o: ouput

    def __init__(
        self,
        args,
        shardings: list[list[int]],
        grid: list[int],
        loc=None,
        ip=None,
    ):
        sharding_attr = ArrayAttr.get(
            [
                ArrayAttr.get([IntegerAttr.get(i64(), s) for s in sharding])
                for sharding in shardings
            ]
        )
        grid_attr = DenseI64ArrayAttr.get(grid)
        super().__init__(
            tensors=args, sharding=sharding_attr, grid=grid_attr, loc=loc, ip=ip
        )
        arg_types = []
        for i, arg in enumerate(args):
            memref_type = MemRefType(arg.type)
            shape = list(memref_type.shape)
            for k, s in enumerate(shardings[i]):
                if s >= len(grid) or s < -1:
                    raise ValueError(
                        f"Invalid sharding value {s} for argument {i}, dimension {k}"
                    )
                if s >= 0:
                    grid_value = grid[s]
                    if shape[k] % grid_value != 0:
                        raise ValueError(
                            f"Shape dimension {shape[k]} at arg {i}, dim {k} is not evenly divisible by grid value {grid_value}"
                        )
                    shape[k] = shape[k] // grid_value
            new_type = MemRefType.get(
                shape,
                memref_type.element_type,
                memref_type.layout,
                memref_type.memory_space,
            )
            arg_types.append(new_type)

        block = self.body.blocks.append(*arg_types)
        with InsertionPoint(block):
            yield_([])

    @property
    def block(self):
        """Returns the body (block) of."""
        return self.regions[0].blocks[0]

    @property
    def interfaces(self):
        if GridMapOp.interface_attr in self.attributes:
            attr = self.attributes[GridMapOp.interface_attr].value
            return [i == "i" for i in attr]
        return [None] * len(self.tensors)


class LibKernel:
    link_attr = "link_with"

    def declare(
        kernel_name,
        itypes,
        otypes,
        link_file=None,  # FIXME
        ip=None,
    ):
        func_type = FunctionType.get(itypes, otypes)
        func_op = FuncOp(name=kernel_name, type=func_type, ip=ip)
        func_op.attributes["sym_visibility"] = StringAttr.get("private")
        # file to be linked for this kernel implementation
        func_op.attributes[LibKernel.link_attr] = StringAttr.get(str(link_file))
        return func_op

    def get_link(func_op):
        assert LibKernel.link_attr in func_op.attributes
        return func_op.attributes[LibKernel.link_attr].value
