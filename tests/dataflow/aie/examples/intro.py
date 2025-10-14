# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available


def gemm_v1(M, N, K):
    Ty = int32
    LyA = Layout("S0R")
    LyB = Layout("RS1")
    LyC = Layout("S0S1")

    @df.region()
    def top():
        @df.kernel(mapping=[2, 2])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np_C = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_allclose(C, np_C, atol=1e-5)
    print("PASSED!")


def gemm_v2(M, N, K):
    Ty = int32
    LyA = Layout("RS0")
    LyB = Layout("S0R")
    LyC = Layout("RR")

    @df.region()
    def top():
        pipe = df.pipe(dtype=Ty, shape=(M, N), depth=2)

        @df.kernel(mapping=[2])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk = df.get_pid()
            with allo.meta_if(pk == 0):
                pipe.put(allo.matmul(A, B))
            with allo.meta_else():
                C[:, :] = allo.add(allo.matmul(A, B), pipe.get())

    mod = df.build(top, target="aie")
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np_C = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_allclose(C, np_C, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    M, N, K = 32, 32, 32
    gemm_v1(M, N, K)
    gemm_v2(M, N, K)
