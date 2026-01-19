# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import int32, Stream, ConstExpr
import allo.dataflow as df
import numpy as np
from allo.backend.aie import is_available

Ty = int32
M, N, K = 16, 16, 16


@df.region()
def top(A: Ty[M, N], B: Ty[M, N]):
    pipe: Stream[Ty, 4]
    pipe1: Stream[Ty, 4]

    @df.kernel(mapping=[2], args=[A])
    def producer(local_A: Ty[M, N]):
        pid = df.get_pid()
        c0: ConstExpr[int32] = 0
        with allo.meta_if(pid == c0):
            for i, j in allo.grid(M, N):
                # load data
                out: Ty = local_A[i, j]
                # send data
                pipe.put(out)
        with allo.meta_elif(pid == 1):
            pipe1.put(local_A[0, 0])

    @df.kernel(mapping=[2], args=[B])
    def consumer(local_B: Ty[M, N]):
        pid = df.get_pid()
        c0: ConstExpr[int32] = 0
        with allo.meta_if(pid == c0):
            for i, j in allo.grid(M, N):
                # receive data
                data = pipe.get()
                # computation
                local_B[i, j] = data + 1
        with allo.meta_else():
            local_B[0, 0] = pipe1.get()


def test_producer_consumer():
    A = np.random.randint(0, 64, (M, K)).astype(np.int32)
    B = np.zeros((M, N), dtype=np.int32)

    sim_mod = df.build(top, target="simulator")
    sim_mod(A, B)
    np.testing.assert_allclose(B, A + 1)
    print("Dataflow Simulator Passed!")


if __name__ == "__main__":
    test_producer_consumer()
