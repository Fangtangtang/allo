# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout

LyW1 = Layout("RS0")
LyW2 = Layout("S0R")


def _test_tensor_parallelism():
    Ty = int32
    M, K, N, L = 8, 8, 8, 8
    P0 = 2
    Nt = N // P0

    @df.region()
    def top():
        Y = df.array(df.pipe(dtype=Ty, shape=(M, Nt), depth=2), shape=(P0,))
        part_Z = df.array(df.pipe(dtype=Ty, shape=(M, L), depth=2), shape=(P0,))

        @df.kernel(mapping=[P0])
        def gemm0(X: Ty[M, K], W1: Ty[K, N] @ LyW1):
            pn = df.get_pid()
            Y[pn].put(allo.matmul(X, W1))

        @df.kernel(mapping=[P0])
        def gemm1(W2: Ty[N, L] @ LyW2):
            pn = df.get_pid()
            part_Z[pn].put(allo.matmul(Y[pn].get(), W2))

        @df.kernel(mapping=[1])
        def acc(Z: Ty[M, L]):
            Z_out: Ty[M, L] = 0
            with allo.meta_for(P0) as i:
                Z_out[:, :] += part_Z[i].get()
            Z[:, :] = Z_out
    mod = df.build(top, target="aie-mlir")

LyA = Layout("S0R")
LyB = Layout("RS1")
LyC = Layout("S0S1")


def _test_gemm_1D():
    Ty = int32
    M, N, K = 16, 16, 16
    P0 = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N], C: Ty[M, N] @ LyA):
            C[:, :] = allo.matmul(A, B)

    df.build(top, target="aie-mlir")
    # mod = df.build(top, target="aie-mlir")
    # A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    # B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    # C = np.zeros((M, N)).astype(np.int32)
    # mod(A, B, C)
    # np.testing.assert_allclose(C, A @ B, atol=1e-5)
    # print("PASSED!")


def _test_gemm_2D():
    TyI, TyO = int32, int32
    M, N, K = 16, 16, 16
    P0, P1 = 4, 4

    @df.region()
    def top():
        @df.kernel(mapping=[P0, P1])
        def gemm(A: TyI[M, K] @ LyA, B: TyI[K, N] @ LyB, C: TyO[M, N] @ LyC):
            C[:, :] = allo.matmul(A, B)

    df.build(top, target="aie-mlir")
    # mod = df.build(top, target="aie-mlir")
    # A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    # B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    # C = np.zeros((M, N)).astype(np.int32)
    # mod(A, B, C)
    # np.testing.assert_allclose(C, A @ B, atol=1e-5)
    # print("PASSED!")


if __name__ == "__main__":
    _test_tensor_parallelism()
    # _test_gemm_1D()
    # _test_gemm_2D()
    # _test_gemm_1D_i16_i16()
    # _test_gemm_2D_i16_i32()
