# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, Stream
import allo.dataflow as df
import numpy as np
from allo.backend.aie import is_available
from allo.memory import Layout

S = Layout.Shard
R = Layout.Replicate


def test_atb_v1():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2
    Ma = M // P0

    @df.region()
    def top(B: Ty[K, N], A0: Ty[Ma, K], A1: Ty[Ma, K], C: Ty[M, N]):
        pipeB: Stream[Ty[K, N], 1][P0]
        pipeC: Stream[Ty[Ma, N], 1][P0]

        @df.kernel(mapping=[1], args=[B])
        def loadB(local_B: Ty[K, N]):
            b = local_B
            pipeB[0].put(b)
            pipeB[1].put(b)

        @df.kernel(mapping=[1], args=[A0])
        def gemm0(local_A: Ty[Ma, K]):
            c0 = allo.matmul(local_A, pipeB[0].get())
            pipeC[0].put(c0)

        @df.kernel(mapping=[1], args=[A1])
        def gemm1(local_A: Ty[Ma, K]):
            c1 = allo.matmul(local_A, pipeB[1].get())
            pipeC[1].put(c1)

        @df.kernel(mapping=[1], args=[C])
        def store(local_C: Ty[M, N]):
            local_C[:Ma, :] = pipeC[0].get()
            local_C[Ma:, :] = pipeC[1].get()

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    if is_available():
        mod = df.build(top, target="aie")
        mod(B, A[:Ma, :], A[Ma:, :], C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_atb_v2():
    Ty = int16
    M, N, K = 16, 16, 16
    P0 = 2
    Ma = M // P0

    @df.region()
    def top(B: Ty[K, N], A: Ty[M, K], C: Ty[M, N]):
        pipeB: Stream[Ty[K, N], 1][P0]
        pipeC: Stream[Ty[Ma, N], 1][P0]

        @df.kernel(mapping=[1], args=[B])
        def loadB(local_B: Ty[K, N]):
            b = local_B
            pipeB[0].put(b)
            pipeB[1].put(b)

        @df.kernel(mapping=[P0], args=[A])
        def gemm(local_A: Ty[M, K] @ [S(0), R]):
            pk = df.get_pid()
            c = allo.matmul(local_A, pipeB[pk].get())
            pipeC[pk].put(c)

        @df.kernel(mapping=[1], args=[C])
        def store(local_C: Ty[M, N]):
            local_C[:Ma, :] = pipeC[0].get()
            local_C[Ma:, :] = pipeC[1].get()

    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    if is_available():
        mod = df.build(top, target="aie")
        mod(B, A, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_atb_v1()
    test_atb_v2()
