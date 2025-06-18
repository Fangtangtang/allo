# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int16, int32
import allo.dataflow as df
import numpy as np
from allo.memory import Layout


def _test_vector_scalar_add():
    Ly = Layout("S0")
    # https://github.com/Xilinx/mlir-aie/tree/main/programming_examples/basic/vector_scalar_add
    Ty = int32
    M = 1024

    @df.region()
    def top():
        @df.kernel(mapping=[2])
        def core(A: Ty[M] @ Ly, B: Ty[M] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    mod = df.build(
        top, target="aie-mlir", mapping_primitives=[("bundle", ["core_0", "core_1"])]
    )
    B = np.zeros(M).astype(np.int32)
    mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("PASSED!")


def _test_producer_consumer():
    Ty = int32
    M, N, K = 16, 16, 16

    @df.region()
    def top():
        pipe = df.pipe(dtype=Ty, shape=(M, N), depth=1)

        @df.kernel(mapping=[1])
        def producer(A: Ty[M, N]):
            pipe.put(A)

        @df.kernel(mapping=[1])
        def consumer(B: Ty[M, N]):
            data = pipe.get()
            for i, j in allo.grid(M, N):
                # computation
                B[i, j] = data[i, j] + 1

    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[("chain", ["producer_0", "consumer_0"])],
    )
    mod(A, B)
    np.testing.assert_allclose(A + 1, B, atol=1e-5)
    print("Passed!")


def _test_pingpong_gemm():

    Ty = int16
    M, N, K = 32, 32, 32
    Pm, Pn, Pk = 2, 2, 2
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_1", "gemm_1_0_1"]),
            ("chain", ["gemm_0_1_0", "gemm_1_1_0"]),
            ("chain", ["gemm_0_1_1", "gemm_1_1_1"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def _test_pingpong_gemm_v1():

    Ty = int16
    M, N, K = 32, 32, 128
    Pm, Pn, Pk = 1, 1, 4
    Mt, Nt, Kt = M // Pm, N // Pn, K // Pk

    LyA = Layout("S1S2")
    LyB = Layout("S2S0")
    LyC = Layout("S1S0")

    @df.region()
    def top():
        pipe = df.array(
            df.pipe(dtype=Ty, shape=(Mt, Nt), depth=2), shape=(Pk - 1, Pm, Pn)
        )

        @df.kernel(mapping=[Pk, Pm, Pn])
        def gemm(A: Ty[M, K] @ LyA, B: Ty[K, N] @ LyB, C: Ty[M, N] @ LyC):
            pk, pm, pn = df.get_pid()
            with allo.meta_if(pk > 0):
                C_in: Ty[Mt, Nt] = pipe[pk - 1, pm, pn].get()
            with allo.meta_else():
                C_in: Ty[Mt, Nt] = 0
            C_out: Ty[Mt, Nt] = allo.add(allo.matmul(A, B), C_in)
            with allo.meta_if(pk < Pk - 1):
                pipe[pk, pm, pn].put(C_out)
            with allo.meta_elif(pk == Pk - 1):
                C[:, :] = C_out

    mod = df.build(
        top,
        target="aie-mlir",
        mapping_primitives=[
            ("chain", ["gemm_0_0_0", "gemm_1_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0", "gemm_2_0_0"]),
            ("chain", ["gemm_0_0_0-gemm_1_0_0-gemm_2_0_0", "gemm_3_0_0"]),
        ],
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int16)
    B = np.random.randint(0, 64, (K, N)).astype(np.int16)
    C = np.zeros((M, N)).astype(np.int16)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


if __name__ == "__main__":
    _test_vector_scalar_add()
    _test_producer_consumer()
    _test_pingpong_gemm()
    _test_pingpong_gemm_v1()
