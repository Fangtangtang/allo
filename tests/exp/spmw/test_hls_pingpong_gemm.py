# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.ir.types import float32, Stream
from allo import spmw
from allo.exp import build


def test_cooperative_gemm():
    Ty = float32
    M, N, K = 16, 16, 16
    P0, P1 = 2, 2
    Mt, Nt = M // P0, N // P1

    @spmw.unit()
    def top(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
        pipe: Stream[Ty[Mt, Nt], 2][P0, P1]

        @spmw.work(grid=[P0, P1])
        def gemm0():
            x, y = spmw.axes()
            pi, pj = x.id, y.id
            C_out: Ty[Mt, Nt] = 0
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    c: Ty = 0
                    for k in range(K // 2):
                        c += A[i, k] * B[k, j]
                    C_out[i - pi * Mt, j - pj * Nt] = c
            pipe[pi, pj].put(C_out)

        @spmw.work(grid=[P0, P1])
        def gemm1():
            x, y = spmw.axes()
            pi, pj = x.id, y.id
            C_out: Ty[Mt, Nt] = pipe[pi, pj].get()
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    c: Ty = 0
                    for k in range(K // 2, K):
                        c += A[i, k] * B[k, j]
                    C[i, j] = C_out[i - pi * Mt, j - pj * Nt] + c

    build(top)


if __name__ == "__main__":
    test_cooperative_gemm()
