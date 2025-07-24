# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# use export USE_VECTORIZED_MATMUL=1 to use vectorized kernel

import os
import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_2D():
    TyI, TyO = int16, int16
    M, N, K = 32, 32, 32
    P0, P1 = 2, 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(
        top,
        target="aie-mlir",
        profile=True,
        warmup=200,
        num_iters=1000,
        use_default_codegen=True,
        trace_size=8192 * 2048,
        trace=[("gemm", (0, 0)),("gemm", (1, 0)),("gemm", (0, 1)),("gemm", (1, 1)),]
    )
    A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_gemm_2D()
