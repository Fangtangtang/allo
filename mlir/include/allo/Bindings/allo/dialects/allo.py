# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from ._allo_ops_gen import *
from .._mlir_libs._allo.allo import *
from .func import FuncOp
from ..ir import FunctionType, StringAttr

class LibKernel:
    link_attr = "link_with"

    def declare(
        kernel_name,
        itypes,
        otypes,
        link_file=None, # FIXME
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