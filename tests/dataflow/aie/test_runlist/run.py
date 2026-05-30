# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=c-extension-no-member, too-many-instance-attributes, too-many-nested-blocks, no-name-in-module, unsupported-binary-operation

"""
aiecc --xclbin-kernel-name=VADD  --xclbin-kernel-id=0x901 --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-name=../test_runlist/vadd.xclbin --no-xchesscc --no-xbridge --peano $PEANO_INSTALL_DIR --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir

aiecc --xclbin-kernel-name=VSUB  --xclbin-kernel-id=0x902 --alloc-scheme=basic-sequential --aie-generate-xclbin --no-compile-host --xclbin-input=../test_runlist/vadd.xclbin --xclbin-name=../test_runlist/final.xclbin --no-xchesscc --no-xbridge --peano $PEANO_INSTALL_DIR --aie-generate-npu-insts --npu-insts-name=insts.txt top.mlir

"""

from pathlib import Path
import subprocess
import numpy as np
from allo.ir.types import int32
from allo.backend.aie.utils import read_tensor_from_file

cmd = "cmake . -DTARGET_NAME=top -DMLIR_AIE_DIR=$RUNTIME_LIB_DIR/.. && cmake --build . --config Release"
with subprocess.Popen(cmd, shell=True) as process:
    process.wait()
if process.returncode != 0:
    raise RuntimeError("Failed to build the MLIR-AIE project")
    
input0 = np.random.randint(0, 100, 256).astype(np.int32)
with (Path("input0.data")).open("wb") as f:
    f.write(input0.tobytes())

cmd = "./top -x final.xclbin -i insts.txt -k MLIR_AIE"
with subprocess.Popen(cmd, shell=True) as process:
    process.wait()
if process.returncode != 0:
    raise RuntimeError("Failed to execute AIE code.")

result = read_tensor_from_file(
    int32,
    (256,),
    Path("output1.data"),
)
print(input0)
print(result)