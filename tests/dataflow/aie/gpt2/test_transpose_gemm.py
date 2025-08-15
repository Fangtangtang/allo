# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int16, int32, float32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("S1R")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int16
    M, N, K = 16, 16, 8
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[N, K], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B.T)

    mod = df.build(top, target="aie-mlir")
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (N, K)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")

if __name__ == "__main__":
    _test_gemm_1D()
