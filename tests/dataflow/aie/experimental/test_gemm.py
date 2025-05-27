# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


TyI, TyO = int16, int32
M, N, K = 128, 64, 32
P0, P1 = 4, 4

@df.region()
def top1():
    @df.kernel(mapping=[P0, P1])
    def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
        C[:, :] = allo.matmul(A, B)

@df.region()
def top2():
    @df.kernel(mapping=[P0])
    def core(A: TyO[M, N] @ LyA, B: TyO[M, N] @ LyA, C: TyO[M, N] @ LyA):
        C[:, :] = allo.add(A, B)

mod1 = df.build(top1, target="aie-mlir",project="top1.prj")
mod2 = df.build(top2, target="aie-mlir",project="top2.prj")

A = np.random.randint(0, 32, (M, K)).astype(np.int16)
B = np.random.randint(0, 32, (K, N)).astype(np.int16)
C_tmp = np.zeros((M, N)).astype(np.int32)
C = np.zeros((M, N)).astype(np.int32)
mod1(A, B, C_tmp)
mod2(C,C_tmp,C)
np.testing.assert_allclose(C, A @ B, atol=1e-5)
print("PASSED!")

