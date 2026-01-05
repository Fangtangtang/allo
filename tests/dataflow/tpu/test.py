# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import allo
from allo.ir.types import float32, Stream
import allo.dataflow as df
import allo.backend.hls as hls
import numpy as np

Ty = float32
M = 16


@df.region()
def top(A: Ty[M], B: Ty[M]):
    pipe: Stream[Ty, 4]

    @df.kernel(mapping=[1], args=[A])
    def producer(local_A: Ty[M]):
        for i in allo.grid(M):
            # load data
            out: Ty = local_A[i]
            # send data
            pipe.put(out)

    @df.kernel(mapping=[1], args=[B])
    def consumer(local_B: Ty[M]):
        for i in allo.grid(M):
            # receive data
            data = pipe.get()
            # computation
            local_B[i] = data + 1


def test_producer_consumer():
    A = np.random.rand(M).astype(np.float32)
    B = np.zeros((M), dtype=np.float32)

    if hls.is_available("vitis_hls"):
        mod = df.build(
            top,
            target="vitis_hls",
            mode="hw_emu",
        )
        mod(A, B)
        np.testing.assert_allclose(A + 1, B)
        print("Passed!")


if __name__ == "__main__":
    test_producer_consumer()
