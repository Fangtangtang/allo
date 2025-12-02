# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import allo
from allo.ir.types import int32, Stream
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


if __name__ == "__main__":
    # test_transfer()
    test_vector_scalar_add()
