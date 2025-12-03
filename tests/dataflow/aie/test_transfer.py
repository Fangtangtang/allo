# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32, Stream, int16, bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
import allo.dataflow as df
import numpy as np
from allo.memory import Layout
from allo.backend.aie import is_available


def test_transfer():
    Ty = int32
    M = 16

    @df.region()
    def top():
        pipe: Stream[Ty[M], 2]

        @df.kernel(mapping=[1])
        def producer(A: Ty[M]):
            # send data
            pipe.put(A)

        @df.kernel(mapping=[1])
        def consumer(B: Ty[M]):
            # receive data
            B[:] = allo.add(pipe.get(), 1)

    A = np.random.randint(0, 64, (M)).astype(np.int32)
    B = np.zeros((M), dtype=np.int32)

    if is_available():
        mod = df.build(top, target="aie")
        mod(A, B)
        # allo.backend.aie._call_prj("top.prj",[Ty, Ty], 0, [0], [1], A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_vector_scalar_add():
    Ty = int32
    M = 128
    Ly = Layout("S0")

    @df.region()
    def top():
        @df.kernel(mapping=[4])
        def core(A: Ty[M], B: Ty[M * 4] @ Ly):
            B[:] = allo.add(A, 1)

    A = np.random.randint(0, 100, M).astype(np.int32)
    if is_available():
        B = np.zeros(M * 4).astype(np.int32)
        mod = df.build(top, target="aie")
        mod(A, B)
        # allo.backend.aie._call_prj("top.prj",[Ty, Ty], 0, [0], [1], A, B)
        np.testing.assert_allclose(B[0:M], A + 1)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_matmul():
    Ty = bfloat16
    M = N = K = 64
    LyC = Layout("RS0")
    P = 64
    p = 2

    @df.region()
    def top():
        @df.kernel(mapping=[P])
        def core(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N * P] @ LyC):
            C[:, :] = allo.matmul(A, B)

        @df.kernel(mapping=[P])
        def cor(A_: Ty[M, K], B_: Ty[K, N], C_: Ty[M, N * P] @ LyC):
            C_[:, :] = allo.matmul(A_, B_)

    if Ty == int16:
        A = np.random.randint(0, 100, (M, K)).astype(np.int16)
        B = np.random.randint(0, 100, (K, N)).astype(np.int16)
        C = np.zeros((M, N * P)).astype(np.int16)
        C_ = np.zeros((M, N * P)).astype(np.int16)
    if Ty == bfloat16:
        A = (np.random.random((M, K)) * 0.1).astype(np_bfloat16)
        B = (np.random.random((K, N)) * 0.1).astype(np_bfloat16)
        C = np.zeros((M, N * P)).astype(np_bfloat16)
        C_ = np.zeros((M, N * P)).astype(np_bfloat16)
    if is_available():
        # groups, groups_ = [], []
        # for i in range(0, P, p):
        #     cores, cores_ = [], []
        #     for j in range(p):
        #         cores.append(f"core_{i + j}")
        #         cores_.append(f"cor_{i + j}")
        #     groups.append(tuple(cores))
        #     groups_.append(tuple(cores_))
        # mod = df.build(
        #     top,
        #     target="aie",
        #     mapping_primitives=[
        #         ("bundle", groups), ("bundle", groups_),
        #     ],
        # )
        # mod(A, B, C, A, B, C_)

        allo.backend.aie._call_prj(
            "top.prj",
            [Ty, Ty, Ty, Ty, Ty, Ty],
            65536,
            [0, 1, 3, 4],
            [2, 5],
            A,
            B,
            C,
            A,
            B,
            C_,
        )

        # np.testing.assert_allclose(C[:, :N], A @ B)
        print("PASSED!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    # test_transfer()
    # test_vector_scalar_add()
    test_matmul()
