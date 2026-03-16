# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.exp import build
from allo.ir.types import int32, ConstExpr, index
from allo import spmw


def test_get_wid_1D_1():
    @spmw.unit()
    def top(A: int32[1024], B: int32[1024]):
        @spmw.work(grid=[1])
        def core():
            for i in range(1024):
                B[i] = A[i] + 1

    build(top)


def test_get_wid_1D_2():
    vlen = 1024
    P = 4
    tlen = vlen // P

    @spmw.unit()
    def top(A: int32[vlen], B: int32[vlen]):
        @spmw.work(grid=[P])
        def core():
            x = spmw.axes()
            pi: ConstExpr[index] = x.id
            for i in range(tlen * pi, tlen * (pi + 1)):
                B[i] = A[i] + 1

    build(top)


if __name__ == "__main__":
    test_get_wid_1D_1()
    test_get_wid_1D_2()
