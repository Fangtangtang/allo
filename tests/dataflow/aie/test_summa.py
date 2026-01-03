# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int16, int32, Stream
from allo.memory import Layout
import numpy as np
from allo.backend.aie import is_available


def test_summa_2x2():
    Ty = int32
    M, K, N = 32, 64, 64
    P0, P1 = 2, 2

    La = Layout("RS1")
    Lb = Layout("S1S0")

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        row_fifo: Stream[Ty[M, N // 2], 1][P0]
        final_fifo: Stream[Ty[M, N], P0][P0]

        @df.kernel(mapping=[P0, P1], args=[A, B])
        def summa(local_A: Ty[M, K] @ La, local_B: Ty[K, N] @ Lb):
            i, j = df.get_pid()
            with allo.meta_if(j == 1):
                row_fifo[i].put(allo.matmul(local_A, local_B))
            with allo.meta_else():
                right_half: Ty[M, N // 2] = row_fifo[i].get()
                F_tile: Ty[M, N] = 0
                P_tile: Ty[M, N // 2] = allo.matmul(local_A, local_B)
                with allo.meta_for(M) as m:
                    with allo.meta_for(N // 2) as n:
                        F_tile[m, n] = P_tile[m, n]
                        F_tile[m, n + N // 2] = right_half[m, n]
                final_fifo[i].put(F_tile)

        @df.kernel(mapping=[1], args=[C])
        def write_c(local_C: Ty[M, N]):
            local_C[:, :] = final_fifo[0].get() + final_fifo[1].get()

    A = np.random.randint(-8, 8, (M, K)).astype(np.int32)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int32)
    C = np.zeros((M, N), dtype=np.int32)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_summa():
    Ty = int16
    M, K, N = 32, 128, 128
    P0, P1 = 4, 4

    La = Layout("RS0")
    Lb = Layout("S0S1")
    Lc = Layout("RS1")

    @df.region()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        column_fifo: Stream[Ty[M, N // P0], 1][P0, P1 - 1]

        @df.kernel(mapping=[P0, P1], args=[B, A, C])
        def summa(
            local_B: Ty[K, N] @ Lb, local_A: Ty[M, K] @ La, local_C: Ty[M, N] @ Lc
        ):
            i, j = df.get_pid()

            with allo.meta_if(j == 0):
                tmp: Ty[M, N // P0] = column_fifo[i, j].get()
                local_C[:, :] = allo.add(allo.matmul(local_A, local_B), tmp)
            with allo.meta_elif(j == P1 - 1):
                column_fifo[i, j - 1].put(allo.matmul(local_A, local_B))
            with allo.meta_else():
                tmp: Ty[M, N // P0] = column_fifo[i, j].get()
                column_fifo[i, j - 1].put(allo.add(allo.matmul(local_A, local_B), tmp))

    A = np.random.randint(-8, 8, (M, K)).astype(np.int16)
    B = np.random.randint(-8, 8, (K, N)).astype(np.int16)
    C = np.zeros((M, N), dtype=np.int16)
    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B, C)
        np.testing.assert_allclose(C, A @ B, atol=1e-5)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_summa_2x2()
    test_summa()
