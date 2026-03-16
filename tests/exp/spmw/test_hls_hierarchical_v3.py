# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import build
import allo
from allo.ir.types import int32, float32, ConstExpr
from allo import spmw


def test1():
    @spmw.unit()
    def vadd[L](A: int32[L], B: int32[L]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(L):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd[1024](A, B)

    build(top)


def test2():
    M, N, K = 32, 32, 32

    @spmw.unit()
    def inner[P0, P1](A: float32[M, K], B: float32[K, N], C: float32[M, N]):
        @spmw.work(grid=[P0, P1])
        def gemm():
            x, y = spmw.axes()
            pi, pj = x.id, y.id
            Mt: ConstExpr[int32] = M // P0
            Nt: ConstExpr[int32] = N // P1
            for i in range(pi * Mt, (pi + 1) * Mt):
                for j in range(pj * Nt, (pj + 1) * Nt):
                    for k in range(K):
                        C[i, j] += A[i, k] * B[k, j]

    @spmw.unit()
    def top(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
        @spmw.work(grid=[2])
        def wrapper():
            x = spmw.axes()
            if x.id == 0:
                inner[2, 2](A, B, C1)
            else:
                inner[4, 4](A, B, C2)

    build(top)

    @spmw.unit()
    def top2(A: float32[M, K], B: float32[K, N], C1: float32[M, N], C2: float32[M, N]):
        @spmw.work(grid=[2])
        def wrapper():
            x = spmw.axes()
            if x.id == 0:
                inner[2, 2](A, B, C1)
            else:
                inner[2, 2](A, B, C2)

    build(top2)


if __name__ == "__main__":
    test1()
    test2()
