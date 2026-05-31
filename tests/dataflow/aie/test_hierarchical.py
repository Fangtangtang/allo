# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import allo
from allo.ir.types import int32, Stream
import allo.dataflow as df
from allo.memory import Layout
from allo.backend.aie import is_available

S = Layout.Shard
R = Layout.Replicate


def test_sequential_vadds():
    M = 1024
    Ty = int32

    P = 4
    Ly = [S(0)]

    @df.region()
    def vadd(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[P], args=[A, B])
        def core1(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            local_B[:] = allo.add(local_A, 1)

    @df.region()
    def vsub(A: Ty[M], B: Ty[M]):
        @df.kernel(mapping=[P], args=[A, B])
        def core2(local_A: Ty[M] @ Ly, local_B: Ty[M] @ Ly):
            local_B[:] = allo.sub(local_A, 1)

    @df.region()
    def top(In: Ty[M], Out: Ty[M]):
        s: Stream[Ty[M], 2]

        @df.kernel(mapping=[1], args=[In])
        def vadd1(A: Ty[M]):
            intermediate: Ty[M]
            vadd(A, intermediate)
            s.put(intermediate)

        @df.kernel(mapping=[1], args=[Out])
        def vadd2(B: Ty[M]):
            intermediate: Ty[M] = s.get()
            vsub(intermediate, B)

    if is_available():
        mod = df.build(top, target="aie")
        A = np.random.randint(0, 100, M).astype(np.int32)
        B = np.zeros(M).astype(np.int32)
        mod(A, B)
        np.testing.assert_allclose(B, A)


if __name__ == "__main__":
    test_sequential_vadds()
