# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
import allo.dataflow as df
from allo.ir.types import int32, Stream
from allo.memory import Layout
import numpy as np
from allo.backend.aie.external_kernel import ExternalModule

# RRxRS->RS
# RSxSR->RR
LyW1 = Layout("RS0")
LyW2 = Layout("S0R")

event0 = ExternalModule(
    top="mark_start",
    impl_path="../../../allo/library/aie/event.cc",
    input_idx=[],
    output_idx=[],
)

event1 = ExternalModule(
    top="mark_end",
    impl_path="../../../allo/library/aie/event.cc",
    input_idx=[],
    output_idx=[],
)


def profiling():
    Ty = int32
    M, N, K = 32, 32, 32

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def gemm(A: Ty[M, K], B: Ty[K, N], C: Ty[M, N]):
            event0()
            C[:, :] = allo.add(A, B)
            event1()

    mod = df.build(
        top, target="aie", profile=True, trace=[("gemm", (0,))], trace_size=65536
    )
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.random.randint(0, 64, (K, N)).astype(np.int32)
    C = np.zeros((M, N)).astype(np.int32)
    mod(A, B, C)
    np.testing.assert_allclose(C, A @ B, atol=1e-5)
    print("PASSED!")


def test():
    Ty = int32
    M, N = 6, 8
    P0 = 3

    @df.region()
    def top():
        @df.kernel(mapping=[P0])
        def core(In: Ty[M, N], Out0: Ty[M, N], Out1: Ty[M, N], Out2: Ty[M, N]):
            pn = df.get_pid()
            with allo.meta_if(pn == 0):
                Out0[:, :] = In
            with allo.meta_elif(pn == 1):
                Out1[:, :] = allo.add(In, 1)
            with allo.meta_else():
                Out2[:, :] = allo.add(In, 2)

    A = np.random.randint(0, 100, (M, N)).astype(np.int32)

    mod = df.build(top, target="aie")
    B0 = np.zeros((M, N)).astype(np.int32)
    B1 = np.zeros((M, N)).astype(np.int32)
    B2 = np.zeros((M, N)).astype(np.int32)
    mod(A, B0, B1, B2)
    np.testing.assert_allclose(B0, A)
    np.testing.assert_allclose(B1, A + 1)
    np.testing.assert_allclose(B2, A + 2)
    print("PASSED!")


if __name__ == "__main__":
    # profiling()
    test()
