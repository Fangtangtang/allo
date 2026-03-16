# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import build
import allo
from allo.ir.types import int32
from allo import spmw


def test():
    @spmw.unit()
    def vadd(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in allo.grid(1024):
                B[i] = A[i] + 1

    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            vadd(A, B)

    build(top)


if __name__ == "__main__":
    test()
