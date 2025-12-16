# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import filelock
from air.backend.xrt import XRTBackend
from air.ir import Context, Location, Module


def _call_prj(project: str, output_idx: list[int], *args):
    """
    TODO: implement execution logic to compile the top.mlir under project dir, compile it and run on the backend.
        args include module inputs and outputs.
    """
    backend = XRTBackend()
    with open(f"{project}/top.mlir", "r") as f:
        content = f.read()

    with Context() as ctx, Location.unknown():
        mlir_module = Module.parse(content)

    compiled_module = backend.compile(mlir_module)
    with filelock.FileLock("/tmp/npu.lock"):
        module_function = backend.load(compiled_module)
        actual_outputs = module_function(*args)

    backend.unload()
    for idx in output_idx:
        args[idx][:] = actual_outputs[idx]
