# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import build
import allo
from allo.ir.types import int32, Stream
from allo import spmw


def test_scalar_stream_1():
    @spmw.unit()
    def top1(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(grid=[1])
        def producer():
            pipe.put(A[0, 0])

        @spmw.work(grid=[1])
        def consumer():
            B[0, 0] = pipe.get()

    build(top1)


def test_scalar_stream_2():
    @spmw.unit()
    def top2(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32]

        @spmw.work(grid=[1])
        def producer():
            for i, j in allo.grid(16, 16):
                pipe.put(A[i, j])

        @spmw.work(grid=[1])
        def consumer():
            for i, j in allo.grid(16, 16):
                B[i, j] = pipe.get()

    build(top2)


def test_tensor_stream():
    @spmw.unit()
    def top(A: int32[16, 16], B: int32[16, 16]):
        pipe: Stream[int32][16, 16]

        @spmw.work(grid=[1])
        def producer():
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    pipe[i, j].put(A[i, j])

        @spmw.work(grid=[1])
        def consumer():
            with allo.meta_for(16) as i:
                with allo.meta_for(16) as j:
                    B[i, j] = pipe[i, j].get()

    build(top)


if __name__ == "__main__":
    test_scalar_stream_1()
    test_scalar_stream_2()
    test_tensor_stream()
