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
        # mod = df.build(top, target="aie")
        # mod(A, B)
        allo.backend.aie._call_prj("top.prj", [Ty, Ty], 0, [0], [1], A, B)
        np.testing.assert_allclose(A + 1, B, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_reuse():
    Ty = int32
    M = 16

    @df.region()
    def top():
        pipe: Stream[Ty[M], 2]

        @df.kernel(mapping=[1])
        def producer(A: Ty[M], B0: Ty[M]):
            B0[:] = allo.add(A, 1)
            # send data
            pipe.put(A)

        @df.kernel(mapping=[1])
        def consumer(B1: Ty[M]):
            # receive data
            B1[:] = allo.add(pipe.get(), 1)

    A = np.random.randint(0, 64, (M)).astype(np.int32)
    B0 = np.zeros((M), dtype=np.int32)
    B1 = np.zeros((M), dtype=np.int32)

    if is_available():
        # mod = df.build(top, target="aie")
        # mod(A, B0, B1)
        allo.backend.aie._call_prj("top.prj", [Ty, Ty, Ty], 0, [0], [1, 2], A, B0, B1)
        np.testing.assert_allclose(A + 1, B0, atol=1e-5)
        np.testing.assert_allclose(A + 1, B1, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


def test_reuse_scale_up():
    Ty = int32
    M = 16
    P = 64
    Ly = Layout("S0")

    @df.region()
    def top():
        pipe: Stream[Ty[M], 2][P]

        @df.kernel(mapping=[P])
        def producer(A: Ty[M * P] @ Ly, B0: Ty[M * P] @ Ly):
            B0[:] = allo.add(A, 1)
            # send data
            p = df.get_pid()
            pipe[p].put(A)

        @df.kernel(mapping=[P])
        def consumer(B1: Ty[M * P] @ Ly):
            p = df.get_pid()
            # receive data
            B1[:] = allo.add(pipe[p].get(), 1)

    A = np.random.randint(0, 64, (M * P,)).astype(np.int32)
    B0 = np.zeros((M * P,), dtype=np.int32)
    B1 = np.zeros((M * P,), dtype=np.int32)

    if is_available():
        groups = []
        for i in range(P):
            groups.append((f"producer_{i}", f"consumer_{i}"))
        mod = df.build(top, target="aie", mapping_primitives=[("bundle", groups)])
        mod(A, B0, B1)

        # allo.backend.aie._call_prj("top.prj", [Ty, Ty, Ty], 65536, [0], [1, 2], A, B0, B1)
        np.testing.assert_allclose(A + 1, B0, atol=1e-5)
        np.testing.assert_allclose(A + 1, B1, atol=1e-5)
        print("Passed!")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    test_transfer()
    test_reuse()
    test_reuse_scale_up()
