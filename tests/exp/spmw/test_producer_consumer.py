# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import float32, Stream
from allo import spmw
from allo.exp import build


def test_producer_consumer():
    Ty = float32
    M, N, K = 16, 16, 16

    @spmw.unit()
    def top(A: Ty[M, N], B: Ty[M, N]):
        pipe: Stream[Ty, 4]

        @spmw.work(grid=[1])
        def producer():
            for i, j in allo.grid(M, N):
                # load data
                out: Ty = A[i, j]
                # send data
                pipe.put(out)

        @spmw.work(grid=[1])
        def consumer():
            for i, j in allo.grid(M, N):
                # receive data
                data = pipe.get()
                # computation
                B[i, j] = data + 1

    build(top)


if __name__ == "__main__":
    test_producer_consumer()
